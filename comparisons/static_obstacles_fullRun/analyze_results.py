import json
import numpy as np

with open('comparison_results.json', 'r') as f:
    data = json.load(f)

print('=== FULL DATASET COMPARISON (941 TRIALS) ===\n')

scenarios_data = data['scenarios']

for scenario_name, scenario_data in scenarios_data.items():
    vga_metrics = np.array([trial['vga_metrics'] for trial in scenario_data])
    drl_metrics = np.array([trial['drl_metrics'] for trial in scenario_data])
    
    n_trials = len(scenario_data)
    
    print(f"{scenario_name}: {n_trials} trials")
    print(f"  VGA Success: {np.mean(vga_metrics[:, 0]):.1%}")
    print(f"  DRL Success: {np.mean(drl_metrics[:, 0]):.1%}")

print('\n=== AGGREGATED METRICS (ALL 941 TRIALS) ===\n')

all_vga = []
all_drl = []
for scenario_name, scenario_data in scenarios_data.items():
    for trial in scenario_data:
        all_vga.append(trial['vga_metrics'])
        all_drl.append(trial['drl_metrics'])

vga_all = np.array(all_vga)
drl_all = np.array(all_drl)

metrics = ['Success', 'SPL', 'Travel Time', 'Collision', 'Jerk', 'Min Clearance', 'Oscillation']

print(f'Total Trials: {len(all_vga)}\n')

for i, name in enumerate(metrics):
    vga_mean = np.mean(vga_all[:, i])
    drl_mean = np.mean(drl_all[:, i])
    
    if name == 'Success':
        print(f'{name}: VGA={vga_mean:.1%} | DRL={drl_mean:.1%}')
    elif name == 'Collision':
        print(f'{name} Rate: VGA={vga_mean:.1%} | DRL={drl_mean:.1%}')
    elif name == 'SPL':
        print(f'{name}: VGA={vga_mean:.3f} | DRL={drl_mean:.3f} (VGA {((vga_mean/drl_mean - 1)*100):+.1f}%)')
    elif name == 'Travel Time':
        print(f'{name}: VGA={vga_mean:.2f}s | DRL={drl_mean:.2f}s (VGA {((vga_mean/drl_mean - 1)*100):+.1f}%)')
    elif name == 'Jerk':
        ratio = drl_mean / vga_mean
        print(f'Avg {name}: VGA={vga_mean:.2f} | DRL={drl_mean:.2f} (DRL {ratio:.1f}x higher)')
    elif name == 'Min Clearance':
        print(f'{name}: VGA={vga_mean:.3f}m | DRL={drl_mean:.3f}m (DRL {((drl_mean/vga_mean - 1)*100):+.1f}%)')
    elif name == 'Oscillation':
        ratio = drl_mean / vga_mean
        print(f'{name} Index: VGA={vga_mean:.3f} | DRL={drl_mean:.3f} (DRL {ratio:.1f}x higher)')

print('\n=== WINNER SUMMARY ===\n')
winners = []
if vga_all[:, 0].mean() > drl_all[:, 0].mean():
    winners.append('VGA: Success Rate')
elif drl_all[:, 0].mean() > vga_all[:, 0].mean():
    winners.append('DRL: Success Rate')
else:
    winners.append('TIE: Success Rate')

if vga_all[:, 1].mean() > drl_all[:, 1].mean():
    winners.append('VGA: SPL')
elif drl_all[:, 1].mean() > vga_all[:, 1].mean():
    winners.append('DRL: SPL')
else:
    winners.append('TIE: SPL')

if vga_all[:, 2].mean() < drl_all[:, 2].mean():
    winners.append('VGA: Travel Time')
elif drl_all[:, 2].mean() < vga_all[:, 2].mean():
    winners.append('DRL: Travel Time')
else:
    winners.append('TIE: Travel Time')

if vga_all[:, 3].mean() < drl_all[:, 3].mean():
    winners.append('VGA: Collision Rate')
elif drl_all[:, 3].mean() < vga_all[:, 3].mean():
    winners.append('DRL: Collision Rate')
else:
    winners.append('TIE: Collision Rate')

if vga_all[:, 4].mean() < drl_all[:, 4].mean():
    winners.append('VGA: Avg Jerk')
elif drl_all[:, 4].mean() < vga_all[:, 4].mean():
    winners.append('DRL: Avg Jerk')
else:
    winners.append('TIE: Avg Jerk')

if vga_all[:, 5].mean() > drl_all[:, 5].mean():
    winners.append('VGA: Min Clearance')
elif drl_all[:, 5].mean() > vga_all[:, 5].mean():
    winners.append('DRL: Min Clearance')
else:
    winners.append('TIE: Min Clearance')

if vga_all[:, 6].mean() < drl_all[:, 6].mean():
    winners.append('VGA: Oscillation Index')
elif drl_all[:, 6].mean() < vga_all[:, 6].mean():
    winners.append('DRL: Oscillation Index')
else:
    winners.append('TIE: Oscillation Index')

for w in winners:
    print(w)

vga_wins = sum('VGA:' in w for w in winners)
drl_wins = sum('DRL:' in w for w in winners)
ties = sum('TIE:' in w for w in winners)

print(f'\nVGA wins: {vga_wins} metrics')
print(f'DRL wins: {drl_wins} metrics')
print(f'Ties: {ties} metrics')
