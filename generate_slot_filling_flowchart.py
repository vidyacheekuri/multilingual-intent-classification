"""
Generate a clean black and white flowchart for XLM-RoBERTa Slot Filling with BIO Tagging
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Black and white color scheme
box_color = 'white'
border_color = 'black'
text_color = 'black'
arrow_color = 'black'

# Title
ax.text(5, 11.5, 'XLM-RoBERTa Slot Filling with BIO Tagging', 
        ha='center', va='center', fontsize=18, fontweight='bold', color=text_color)

# 1. Input Utterance
input_box = FancyBboxPatch((3.5, 9.5), 3, 0.8,
                           boxstyle="round,pad=0.1", 
                           facecolor=box_color, 
                           edgecolor=border_color, 
                           linewidth=2)
ax.add_patch(input_box)
ax.text(5, 10.1, 'Input Utterance', ha='center', va='center', 
        fontsize=12, fontweight='bold', color=text_color)
ax.text(5, 9.7, "'Set alarm at 5 AM tomorrow'", ha='center', va='center', 
        fontsize=10, style='italic', color=text_color)

# Arrow 1: Input to Tokenization
arrow1 = FancyArrowPatch((5, 9.5), (5, 9.0), 
                         arrowstyle='->', lw=2, color=arrow_color)
ax.add_patch(arrow1)

# 2. Tokenization
token_box = FancyBboxPatch((3.5, 8.2), 3, 0.8,
                           boxstyle="round,pad=0.1", 
                           facecolor=box_color, 
                           edgecolor=border_color, 
                           linewidth=2)
ax.add_patch(token_box)
ax.text(5, 8.7, 'Tokenization (SentencePiece)', ha='center', va='center', 
        fontsize=12, fontweight='bold', color=text_color)
ax.text(5, 8.3, 'WordPieces + [CLS]/[SEP]', ha='center', va='center', 
        fontsize=9, color=text_color)

# Arrow 2: Tokenization to Encoder
arrow2 = FancyArrowPatch((5, 8.2), (5, 7.7), 
                         arrowstyle='->', lw=2, color=arrow_color)
ax.add_patch(arrow2)

# 3. XLM-RoBERTa Encoder (large box)
encoder_box = FancyBboxPatch((2.5, 6.2), 5, 1.5,
                             boxstyle="round,pad=0.15", 
                             facecolor=box_color, 
                             edgecolor=border_color, 
                             linewidth=2.5)
ax.add_patch(encoder_box)
ax.text(5, 7.4, 'XLM-RoBERTa Encoder', ha='center', va='center', 
        fontsize=13, fontweight='bold', color=text_color)
ax.text(5, 7.0, '12 layers, 12 heads, hidden size 768', ha='center', va='center', 
        fontsize=10, color=text_color)

# Arrow 3: Encoder to Representations
arrow3 = FancyArrowPatch((5, 6.2), (5, 5.75), 
                         arrowstyle='->', lw=2, color=arrow_color)
ax.add_patch(arrow3)

# 4. Token Representations (h1, h2, ..., hn) - positioned to the right
rep_y = 5.5
rep_x_start = 6.5
rep_spacing = 0.5
rep_labels = ['h₁', 'h₂', '...', 'hₙ']

for i, label in enumerate(rep_labels):
    x_pos = rep_x_start + i * rep_spacing
    rep_box = FancyBboxPatch((x_pos - 0.2, rep_y - 0.25), 0.4, 0.5,
                             boxstyle="round,pad=0.05", 
                             facecolor=box_color, 
                             edgecolor=border_color, 
                             linewidth=1.5)
    ax.add_patch(rep_box)
    ax.text(x_pos, rep_y, label, ha='center', va='center', 
            fontsize=10, fontweight='bold', color=text_color)

ax.text(7.5, rep_y - 0.6, '(token representations)', ha='center', va='center', 
        fontsize=9, style='italic', color=text_color)

# Arrow from Encoder to Token Representations (horizontal)
arrow3_h = FancyArrowPatch((7.5, 7.0), (6.5, rep_y), 
                           arrowstyle='->', lw=2, color=arrow_color)
ax.add_patch(arrow3_h)

# Arrow from h_n to BIO layer
arrow3b = FancyArrowPatch((8.3, rep_y), (7.5, 4.8), 
                          arrowstyle='->', lw=1.5, color=arrow_color, 
                          connectionstyle="arc3,rad=0.2")
ax.add_patch(arrow3b)

# 5. BIO Tagging Layer
bio_box = FancyBboxPatch((2.5, 4.0), 5, 0.8,
                         boxstyle="round,pad=0.1", 
                         facecolor=box_color, 
                         edgecolor=border_color, 
                         linewidth=2)
ax.add_patch(bio_box)
ax.text(5, 4.5, 'BIO Tagging Layer', ha='center', va='center', 
        fontsize=12, fontweight='bold', color=text_color)

# Arrow 4: Encoder to BIO Layer
arrow4 = FancyArrowPatch((5, 6.2), (5, 4.8), 
                         arrowstyle='->', lw=2, color=arrow_color)
ax.add_patch(arrow4)

# 6. Linear + CRF
crf_box = FancyBboxPatch((3.0, 2.8), 4, 0.8,
                         boxstyle="round,pad=0.1", 
                         facecolor=box_color, 
                         edgecolor=border_color, 
                         linewidth=2)
ax.add_patch(crf_box)
ax.text(5, 3.3, 'Linear layer + CRF', ha='center', va='center', 
        fontsize=12, fontweight='bold', color=text_color)

# Arrow 5: BIO to CRF
arrow5 = FancyArrowPatch((5, 4.0), (5, 3.6), 
                         arrowstyle='->', lw=2, color=arrow_color)
ax.add_patch(arrow5)

# 7. Slot Label Sequence
label_box = FancyBboxPatch((2.5, 1.5), 5, 0.8,
                           boxstyle="round,pad=0.1", 
                           facecolor=box_color, 
                           edgecolor=border_color, 
                           linewidth=2)
ax.add_patch(label_box)
ax.text(5, 2.0, 'Slot label sequence', ha='center', va='center', 
        fontsize=12, fontweight='bold', color=text_color)
ax.text(5, 1.6, 'O, O, O, O, B-time, I-time, O, B-date', ha='center', va='center', 
        fontsize=9, style='italic', color=text_color, family='monospace')

# Arrow 6: CRF to Labels
arrow6 = FancyArrowPatch((5, 2.8), (5, 2.3), 
                         arrowstyle='->', lw=2, color=arrow_color)
ax.add_patch(arrow6)

# 8. Slot Inventory
inventory_box = FancyBboxPatch((0.5, 0.2), 4, 0.8,
                               boxstyle="round,pad=0.1", 
                               facecolor=box_color, 
                               edgecolor=border_color, 
                               linewidth=2)
ax.add_patch(inventory_box)
ax.text(2.5, 0.7, 'Slot Inventory', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=text_color)
ax.text(2.5, 0.3, '111 labels: 55 B-*, 55 I-*, 1 O', ha='center', va='center', 
        fontsize=9, color=text_color)

# Arrow 7: Labels to Inventory
arrow7 = FancyArrowPatch((2.5, 1.5), (2.5, 1.0), 
                         arrowstyle='->', lw=2, color=arrow_color)
ax.add_patch(arrow7)

# Arrow from Inventory back to BIO Tagging Layer (feedback loop)
arrow_feedback = FancyArrowPatch((0.5, 0.6), (2.5, 4.4), 
                                 arrowstyle='->', lw=1.5, color=arrow_color,
                                 connectionstyle="arc3,rad=0.3", linestyle='--')
ax.add_patch(arrow_feedback)

# 9. Supported Slots
slots_box = FancyBboxPatch((5.5, 0.2), 4, 0.8,
                           boxstyle="round,pad=0.1", 
                           facecolor=box_color, 
                           edgecolor=border_color, 
                           linewidth=2)
ax.add_patch(slots_box)
ax.text(7.5, 0.7, 'Supported Slots', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=text_color)
ax.text(7.5, 0.3, 'Time, location, person, media, food,\ncommunication, business, transport, IoT', 
        ha='center', va='center', fontsize=8, color=text_color)

# Arrow 8: Inventory to Supported Slots
arrow8 = FancyArrowPatch((4.5, 0.6), (5.5, 0.6), 
                         arrowstyle='->', lw=2, color=arrow_color)
ax.add_patch(arrow8)

# Save the figure
plt.tight_layout()
output_path = '/Users/vidyacheekuri/Library/CloudStorage/GoogleDrive-vidyacheekuri.us@gmail.com/My Drive/intent_project/visualizations/slot_filling_flowchart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', 
            edgecolor='none', pad_inches=0.2)
print(f"✓ Clean black and white flowchart saved to: {output_path}")
plt.close()
