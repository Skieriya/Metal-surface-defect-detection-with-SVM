import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skops.io as sio
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

model_path = 'best_model.skops'


untrusted_types = sio.get_untrusted_types(file=model_path)
state = sio.load(model_path, trusted=untrusted_types)

model = state['best_model']
scaler = state['scaler']
le = state['encoder']
class_names = state['classes']
model_name = state.get('best_name', 'Best Model')

preds = model.predict(x_val_scaled)

cm = confusion_matrix(y_val, preds)
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues', ax=ax_cm)
ax_cm.set_ylabel('True Class')
ax_cm.set_xlabel('Predicted Class')
ax_cm.set_title(f'Confusion Matrix - {model_name}')
plt.tight_layout()
plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300)
plt.close(fig_cm)

_, _, f1_per_class, _ = precision_recall_fscore_support(y_val, preds, labels=range(len(class_names)))

fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))
bars = ax_f1.bar(class_names, f1_per_class, color=sns.color_palette("viridis", len(class_names)))
ax_f1.set_ylabel('F1-Score')
ax_f1.set_xlabel('Class Label')
ax_f1.set_title(f'Per-Class F1-Scores - {model_name}')
ax_f1.set_ylim(0, 1.1)
plt.xticks(rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    ax_f1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'f1_scores_{model_name}.png', dpi=300)
plt.close(fig_f1)