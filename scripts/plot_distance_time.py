import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

# Simulate a distance-time curve: monotonically increasing, with varying speed
t = np.linspace(0, 6, 300)
# Piecewise: fast start, slow middle, fast end
distance = np.piecewise(t,
    [t < 2, (t >= 2) & (t < 4), t >= 4],
    [lambda t: 3 * t**1.5,                        # fast phase
     lambda t: 3 * 2**1.5 + 1.2 * (t - 2),        # slow phase
     lambda t: 3 * 2**1.5 + 1.2 * 2 + 4 * (t - 4)**1.5]  # fast again
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t, distance, color='#4FC3F7', linewidth=2.5)

# Highlight the fast and slow intervals
ax.axvspan(0, 2, alpha=0.15, color='#66BB6A', label='High velocity (steep)')
ax.axvspan(2, 4, alpha=0.15, color='#EF5350', label='Low velocity (flat)')
ax.axvspan(4, 6, alpha=0.15, color='#66BB6A')

# Draw Δy/Δx annotations for the fast interval
t1, t2 = 0.5, 1.5
d1 = 3 * t1**1.5
d2 = 3 * t2**1.5
ax.annotate('', xy=(t2, d1), xytext=(t1, d1),
            arrowprops=dict(arrowstyle='<->', color='#FFD54F', lw=1.5))
ax.text((t1 + t2) / 2, d1 - 0.6, r'$\Delta t$', ha='center', fontsize=13, color='#FFD54F')
ax.annotate('', xy=(t2, d2), xytext=(t2, d1),
            arrowprops=dict(arrowstyle='<->', color='#FFD54F', lw=1.5))
ax.text(t2 + 0.2, (d1 + d2) / 2, r'$\Delta d$', ha='left', fontsize=13, color='#FFD54F')

# Draw Δy/Δx annotations for the slow interval
t3, t4 = 2.5, 3.5
d3 = 3 * 2**1.5 + 1.2 * (t3 - 2)
d4 = 3 * 2**1.5 + 1.2 * (t4 - 2)
ax.annotate('', xy=(t4, d3), xytext=(t3, d3),
            arrowprops=dict(arrowstyle='<->', color='#CE93D8', lw=1.5))
ax.text((t3 + t4) / 2, d3 - 0.6, r'$\Delta t$', ha='center', fontsize=13, color='#CE93D8')
ax.annotate('', xy=(t4, d4), xytext=(t4, d3),
            arrowprops=dict(arrowstyle='<->', color='#CE93D8', lw=1.5))
ax.text(t4 + 0.2, (d3 + d4) / 2, r'$\Delta d$', ha='left', fontsize=13, color='#CE93D8')

ax.set_xlabel('Time (t)', fontsize=13)
ax.set_ylabel('Distance (d)', fontsize=13)
ax.set_title('Distance vs Time — Derivative as Velocity', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/saisat/apiboot/apiboot.ideas/static/images/sainageswar/calculus-derivatives-distance-time.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: calculus-derivatives-distance-time.png")
