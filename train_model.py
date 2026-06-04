"""
train_model.py  —  MediBridge v2
=================================
Input : keypoint_data/data.pkl  →  (N, 45, 466)
Output: best_model.keras, label_encoder.pkl

Changes from previous version
──────────────────────────────
1. Input shape is read from data.pkl — no hardcoded (30, 252).
2. Transformer encoder is the default architecture (best accuracy).
3. CNN-LSTM branch now has Dropout layers.
4. Class-weight balancing for uneven per-class sample counts.
5. ReduceLROnPlateau alongside EarlyStopping.
6. Label smoothing 0.1 — reduces overconfident softmax.
7. Per-class precision/recall report after training.
8. model_meta.pkl removed — realtime reads shape from model.input_shape.
"""

import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import LabelEncoder
from sklearn.metrics            import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ── Load data ─────────────────────────────────────────────────────────────────
with open('keypoint_data/data.pkl', 'rb') as f:
    dataset = pickle.load(f)

data        = dataset['data']    # (N, T, F)
labels      = dataset['labels']
SEQ_LEN     = int(data.shape[1])   # 45
FEATURE_DIM = int(data.shape[2])   # 466

print(f'Dataset shape  : {data.shape}')
print(f'Seq length     : {SEQ_LEN}')
print(f'Feature dim    : {FEATURE_DIM}')
print(f'Classes        : {sorted(set(labels))}')

# ── Encode labels ─────────────────────────────────────────────────────────────
le          = LabelEncoder()
labels_int  = le.fit_transform(labels)
num_classes = len(le.classes_)
print(f'Num classes    : {num_classes}  →  {list(le.classes_)}')

labels_onehot = tf.keras.utils.to_categorical(labels_int, num_classes)

# ── Class weights ─────────────────────────────────────────────────────────────
cw_arr  = compute_class_weight('balanced', classes=np.arange(num_classes), y=labels_int)
cw_dict = dict(enumerate(cw_arr))

# ── Train / val split ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    data, labels_onehot,
    test_size=0.2, shuffle=True, stratify=labels_int, random_state=42,
)
print(f'Train samples  : {len(X_train)}')
print(f'Val samples    : {len(X_val)}')

# ── Architecture ─────────────────────────────────────────────────────────────
# 'transformer'  ← default: best accuracy, captures temporal attention
# 'lstm'         ← simpler, good if dataset is small (< 50 samples/class)
# 'cnn_lstm'     ← good middle ground
ARCH = 'transformer'


def build_transformer(seq_len, feat_dim, num_cls,
                      d_model=128, num_heads=4, num_layers=2,
                      ff_dim=256, dropout=0.3):
    inputs = tf.keras.Input(shape=(seq_len, feat_dim), name='sequence_input')
    x      = tf.keras.layers.Dense(d_model, name='input_proj')(inputs)

    positions = tf.range(seq_len)
    pos_emb   = tf.keras.layers.Embedding(seq_len, d_model, name='pos_emb')(positions)
    x = x + pos_emb

    for i in range(num_layers):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            dropout=dropout, name=f'mha_{i}'
        )(x, x)
        x = tf.keras.layers.LayerNormalization(name=f'ln1_{i}')(x + attn)

        ff = tf.keras.layers.Dense(ff_dim, activation='relu', name=f'ff1_{i}')(x)
        ff = tf.keras.layers.Dropout(dropout,  name=f'ff_drop_{i}')(ff)
        ff = tf.keras.layers.Dense(d_model,    name=f'ff2_{i}')(ff)
        x  = tf.keras.layers.LayerNormalization(name=f'ln2_{i}')(x + ff)

    x = tf.keras.layers.GlobalAveragePooling1D(name='gap')(x)
    x = tf.keras.layers.Dropout(dropout, name='head_drop')(x)
    out = tf.keras.layers.Dense(num_cls, activation='softmax', name='classifier')(x)
    return tf.keras.Model(inputs, out, name='transformer_encoder')


if ARCH == 'transformer':
    model = build_transformer(SEQ_LEN, FEATURE_DIM, num_classes)

elif ARCH == 'lstm':
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                             input_shape=(SEQ_LEN, FEATURE_DIM)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.4),          # higher dropout at peak capacity
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ], name='lstm_model')

elif ARCH == 'cnn_lstm':
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, 3, padding='same',
                               input_shape=(SEQ_LEN, FEATURE_DIM)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ], name='cnn_lstm_model')

else:
    raise ValueError(f'Unknown ARCH: {ARCH}')

model.summary()

# ── Compile ───────────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy'],
)

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
   tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=25,
    restore_best_weights=True, verbose=1,
),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_accuracy',
        save_best_only=True, verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=7, min_lr=1e-6, verbose=1,
    ),
]

# ── Train ─────────────────────────────────────────────────────────────────────
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    class_weight=cw_dict,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'\nVal accuracy   : {val_acc * 100:.2f}%')

y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
y_true = np.argmax(y_val, axis=1)
print('\n── Per-class report ─────────────────────────────────────────')
print(classification_report(y_true, y_pred, target_names=le.classes_))

# ── Verify output shape matches what realtime expects ─────────────────────────
assert model.input_shape[1] == SEQ_LEN,     "Saved model seq_len mismatch"
assert model.input_shape[2] == FEATURE_DIM, "Saved model feature_dim mismatch"
print(f'\nModel input verified: {model.input_shape}  ✓')

# ── Save ──────────────────────────────────────────────────────────────────────
model.save('model.keras')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print('\nSaved → best_model.keras')
print('Saved → model.keras')
print('Saved → label_encoder.pkl')