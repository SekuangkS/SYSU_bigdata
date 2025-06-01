import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, ArrowStyle
from matplotlib import rcParams
import numpy as np

# Set professional style
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['mathtext.fontset'] = 'stix'

def draw_system_architecture():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_facecolor('#f8f9fa')
    
    # Define components with pastel colors
    components = {
        'Input Data': {'pos': (0.1, 0.8), 'radius': 0.06, 'color': '#89CFF0'},
        'Recall': {'pos': (0.3, 0.8), 'radius': 0.06, 'color': '#FFB6C1'},
        'Coarse Rank': {'pos': (0.5, 0.8), 'radius': 0.06, 'color': '#FFD700'},
        'Fine Rank': {'pos': (0.7, 0.8), 'radius': 0.06, 'color': '#98FB98'},
        'Re-rank': {'pos': (0.9, 0.8), 'radius': 0.06, 'color': '#FFA07A'},
        'Resume Tower': {'pos': (0.3, 0.5), 'size': (0.2, 0.25), 'color': '#ADD8E6'},
        'Job Tower': {'pos': (0.6, 0.5), 'size': (0.2, 0.25), 'color': '#FFC0CB'},
        'Output': {'pos': (0.9, 0.5), 'radius': 0.06, 'color': '#DDA0DD'}
    }
    
    # Draw circular components
    for name, props in components.items():
        if 'radius' in props:
            x, y = props['pos']
            circle = Circle(props['pos'], props['radius'], 
                          facecolor=props['color'],
                          edgecolor='#2d2d2d',
                          linewidth=1.5,
                          alpha=0.8)
            ax.add_patch(circle)
            ax.text(x, y, name, 
                   ha='center', va='center',
                   color='black', fontsize=10, weight='bold')
    
    # Draw tower components
    for name in ['Resume Tower', 'Job Tower']:
        props = components[name]
        x, y = props['pos']
        w, h = props['size']
        rect = Rectangle((x-w/2, y-h/2), w, h,
                        facecolor=props['color'],
                        edgecolor='#2d2d2d',
                        linewidth=1.5,
                        alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, name, 
               ha='center', va='center',
               color='black', fontsize=10, weight='bold')
    
    # Define connections with labels
    connections = [
        ('Input Data', 'Recall', 'raw data'),
        ('Recall', 'Coarse Rank', 'candidates'),
        ('Coarse Rank', 'Fine Rank', 'top 200'),
        ('Fine Rank', 'Re-rank', 'scores'),
        ('Re-rank', 'Output', 'final recs'),
        ('Recall', 'Resume Tower', 'resumes'),
        ('Recall', 'Job Tower', 'jobs'),
        ('Resume Tower', 'Fine Rank', 'resume vec'),
        ('Job Tower', 'Fine Rank', 'job vec')
    ]
    
    # Draw connections with thick arrows and labels
    for start, end, label in connections:
        start_props = components[start]
        end_props = components[end]
        
        if 'radius' in start_props:
            start_x, start_y = start_props['pos']
        else:
            start_x, start_y = start_props['pos']
            start_y += start_props['size'][1]/2
        
        if 'radius' in end_props:
            end_x, end_y = end_props['pos']
        else:
            end_x, end_y = end_props['pos']
            end_y += end_props['size'][1]/2
        
        # Calculate exact arrow start/end points
        if 'radius' in start_props:
            start_x = start_props['pos'][0] + start_props['radius']
        else:  # For rectangular components
            start_x = start_props['pos'][0] + start_props['size'][0]/2
        
        if 'radius' in end_props:
            end_x = end_props['pos'][0] - end_props['radius']
        else:  # For rectangular components
            end_x = end_props['pos'][0] - end_props['size'][0]/2
        
        # Draw arrow from right edge to left edge
        ax.annotate("",
                   xy=(end_x, end_y),
                   xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.3",
                                 color='#2d2d2d',
                                 linewidth=1.5))
        
        # Add connection label with larger font
        label_x = (start_x + end_x)/2
        label_y = (start_y + end_y)/2 + 0.02  # Shift label up slightly
        ax.text(label_x, label_y, label,
               ha='center', va='center',
               fontsize=10, backgroundcolor='white',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))
    
    # Add formula with better positioning
    formula = r"$P(y=1|r,j) = \sigma(w^T\phi([f_r(r); f_j(j)]))$"
    ax.text(0.5, 0.6, formula, fontsize=12, ha='center',
           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Add process description with better positioning
    ax.text(0.5, 0.9, "Four-stage Recommendation Process", 
           fontsize=12, ha='center', weight='bold')
    ax.text(0.5, 0.87, "Twin-Tower Model for Fine Ranking", 
           fontsize=10, ha='center', style='italic')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('system_architecture.png', bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    draw_system_architecture()
