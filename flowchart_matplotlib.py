"""
Flowchart simple avec matplotlib - génère directement un PNG
Version épurée noir et blanc
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def create_flowchart():
    """
    Crée un flowchart propre avec matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # Style
    box_style = dict(boxstyle='round,pad=0.1', edgecolor='black',
                     facecolor='white', linewidth=2)
    diamond_style = dict(edgecolor='black', facecolor='lightgray', linewidth=2)
    arrow_style = dict(arrowstyle='->', lw=2, color='black')

    def add_box(x, y, w, h, text, style='box'):
        """Ajoute une boîte"""
        if style == 'box':
            box = FancyBboxPatch((x-w/2, y-h/2), w, h, **box_style)
            ax.add_patch(box)
        elif style == 'diamond':
            points = [[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]]
            box = mpatches.Polygon(points, closed=True, **diamond_style)
            ax.add_patch(box)
        elif style == 'ellipse':
            box = mpatches.Ellipse((x, y), w, h,
                                   edgecolor='black', facecolor='lightgray', linewidth=2)
            ax.add_patch(box)

        ax.text(x, y, text, ha='center', va='center', fontsize=9,
               fontweight='bold', wrap=True)

    def add_arrow(x1, y1, x2, y2, label=''):
        """Ajoute une flèche"""
        arrow = FancyArrowPatch((x1, y1), (x2, y2), **arrow_style)
        ax.add_patch(arrow)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.3, my, label, fontsize=8, style='italic')

    # Positions
    x_center = 5
    x_left = 2.5
    x_right = 7.5

    y = 19

    # Titre
    ax.text(x_center, 19.5, 'MÉTHODOLOGIE D\'ANALYSE\nCIRCUITS SUPRACONDUCTEURS',
           ha='center', fontsize=12, weight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # 1. Départ
    add_box(x_center, y, 2.5, 0.8, '1. DÉPART\nCircuit\nsupraconducteur', 'ellipse')
    add_arrow(x_center, y-0.5, x_center, y-1.2)
    y -= 1.5

    # 2. Identification
    add_box(x_center, y, 2.8, 0.8, '2. IDENTIFICATION\nJosephson, C, L')
    add_arrow(x_center, y-0.5, x_center, y-1.2)
    y -= 1.5

    # 3. Analyse classique
    add_box(x_center, y, 2.8, 0.7, '3. ANALYSE CLASSIQUE\nKirchhoff')
    add_arrow(x_center, y-0.45, x_center, y-1.1)
    y -= 1.4

    # 4. Choix base (diamond)
    add_box(x_center, y, 2, 0.9, '4. CHOIX\nBASE?', 'diamond')
    add_arrow(x_center-0.7, y-0.5, x_left, y-1.0, 'Charge')
    add_arrow(x_center+0.7, y-0.5, x_right, y-1.0, 'Phase')
    y -= 1.5

    # 5a. Base charge
    add_box(x_left, y, 2.2, 0.9, '5a. BASE CHARGE\nL(n,φ) → H\nQuantification')

    # 5b. Base phase
    add_box(x_right, y, 2.2, 0.9, '5b. BASE PHASE\nL(φ,φ̇) → H\nQuantification')

    # Merge to matrix
    add_arrow(x_left, y-0.5, x_left, y-1.0)
    add_arrow(x_right, y-0.5, x_right, y-1.0)
    add_arrow(x_left, y-1.0, x_center, y-1.5)
    add_arrow(x_right, y-1.0, x_center, y-1.5)
    y -= 2.0

    # 8. Matrice
    y_matrix = y
    add_box(x_center, y, 3, 0.8, '8. MATRICE HAMILTONIENNE\nTroncature N')
    add_arrow(x_center, y-0.5, x_center, y-1.1)
    y -= 1.4

    # 9. Méthode (diamond)
    add_box(x_center, y, 2.2, 0.9, '9. MÉTHODE?', 'diamond')
    add_arrow(x_center-0.7, y-0.5, x_left, y-1.0, 'Pert.')
    add_arrow(x_center+0.7, y-0.5, x_right, y-1.0, 'Exact')
    y -= 1.5

    # 10a. Perturbatif
    add_box(x_left, y, 2.2, 0.8, '10a. PERTURBATIF\nSchrieffer-Wolff')

    # 10b. Exact
    add_box(x_right, y, 2.2, 0.8, '10b. EXACT\nDiagonalisation')

    # Merge to spectrum
    add_arrow(x_left, y-0.5, x_left, y-1.0)
    add_arrow(x_right, y-0.5, x_right, y-1.0)
    add_arrow(x_left, y-1.0, x_center, y-1.5)
    add_arrow(x_right, y-1.0, x_center, y-1.5)
    y -= 2.0

    # 11. Spectre
    add_box(x_center, y, 2.5, 0.7, '11. SPECTRE\nE₀, E₁, E₂...')
    add_arrow(x_center, y-0.45, x_center, y-1.1)
    y -= 1.4

    # 12. Convergence (diamond)
    add_box(x_center, y, 2.2, 0.9, '12. CONV.?', 'diamond')
    add_arrow(x_center, y-0.55, x_center, y-1.2, 'Oui')
    y -= 1.5

    # 13. Analyse physique
    add_box(x_center, y, 3, 0.8, '13. ANALYSE PHYSIQUE\nTransitions, anharmonicité')
    add_arrow(x_center, y-0.5, x_center, y-1.1)
    y -= 1.4

    # 14. Dépendances
    add_box(x_center, y, 2.8, 0.7, '14. DÉPENDANCES\nΦₑₓₜ, nₘ')
    add_arrow(x_center, y-0.45, x_center, y-1.1)
    y -= 1.4

    # 15. Visualisation
    add_box(x_center, y, 2.8, 0.7, '15. VISUALISATION\nSpectres 1D/2D')
    add_arrow(x_center, y-0.45, x_center, y-1.1)
    y -= 1.4

    # 16. Fin
    add_box(x_center, y, 2.5, 0.8, '16. FIN\nCircuit caractérisé', 'ellipse')

    plt.tight_layout()
    plt.savefig('flowchart_methodology.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("[OK] Flowchart généré: flowchart_methodology.png")
    plt.close()


if __name__ == '__main__':
    print("="*70)
    print("GÉNÉRATION FLOWCHART MATPLOTLIB")
    print("="*70)
    create_flowchart()
    print("Fichier créé: flowchart_methodology.png (haute résolution)")
