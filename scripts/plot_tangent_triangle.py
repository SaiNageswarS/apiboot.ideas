import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

# Curve: f(x) = 0.5*x^2 + 1
x = np.linspace(-1, 5, 300)
y = 0.5 * x**2 + 1

# Point of tangency
a = 2.5
fa = 0.5 * a**2 + 1
dfa = a  # derivative of 0.5*x^2 + 1 is x

# Tangent line
tangent = fa + dfa * (x - a)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color='#4FC3F7', linewidth=2.5, label=r'$f(x) = \frac{1}{2}x^2 + 1$')
ax.plot(x, tangent, '--', color='#FF8A65', linewidth=2, label=f'Tangent at x={a}')
ax.scatter([a], [fa], color='#EF5350', s=80, zorder=5)

# Draw the right triangle on the tangent
dx = 1.5
dy = dfa * dx
x_start = a - 0.3
y_start = fa + dfa * (x_start - a)
x_end = x_start + dx
y_end = y_start + dy

# Triangle edges
ax.plot([x_start, x_end], [y_start, y_start], color='#FFD54F', linewidth=2)  # horizontal
ax.plot([x_end, x_end], [y_start, y_end], color='#FFD54F', linewidth=2)      # vertical

# Labels
ax.text((x_start + x_end) / 2, y_start - 0.4, r'$\Delta x$', ha='center', fontsize=14, color='#FFD54F')
ax.text(x_end + 0.15, (y_start + y_end) / 2, r'$\Delta y$', ha='left', fontsize=14, color='#FFD54F')

# Angle arc
theta = np.degrees(np.arctan(dfa))
arc_angles = np.linspace(0, np.radians(theta), 30)
arc_r = 0.6
arc_x = x_start + arc_r * np.cos(arc_angles)
arc_y = y_start + arc_r * np.sin(arc_angles)
ax.plot(arc_x, arc_y, color='#66BB6A', linewidth=2)
ax.text(x_start + arc_r + 0.1, y_start + 0.25, r'$\theta$', fontsize=14, color='#66BB6A')

# Formula annotation
ax.text(0.5, 8.5, r'$\tan(\theta) = \frac{\Delta y}{\Delta x} = f\prime(x)$',
        fontsize=15, color='white',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#333333', edgecolor='#666666'))

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('f(x)', fontsize=13)
ax.set_title(r'Tangent Line: Slope = $\tan(\theta) = \Delta y / \Delta x$', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(-1, 12)
ax.set_xlim(-1, 5)

plt.tight_layout()
plt.savefig('/home/saisat/apiboot/apiboot.ideas/static/images/sainageswar/calculus-derivatives-tangent-triangle.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: calculus-derivatives-tangent-triangle.png")
