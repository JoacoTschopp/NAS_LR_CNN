#!/usr/bin/env python3
"""
Análisis de logs NAS para demostración del proceso.

Extrae métricas clave y genera resumen del proceso de búsqueda.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict


def parse_nas_log(log_file):
    """Parse NAS log file y extrae métricas clave."""
    
    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extraer información
    info = {
        'episodes': [],
        'architectures': [],
        'rewards': [],
        'layer_changes': [],
        'best_architecture': None,
        'best_reward': 0.0,
    }
    
    # Buscar configuración inicial
    config_match = re.search(r'Total architectures to evaluate: ([\d,]+)', content)
    if config_match:
        info['total_architectures'] = config_match.group(1)
    
    device_match = re.search(r'device: (\w+)', content)
    if device_match:
        info['device'] = device_match.group(1)
    
    # Buscar cambios de capas
    layer_changes = re.findall(
        r'LAYER SCHEDULE: Increasing depth to (\d+) layers \(after (\d+) architectures\)',
        content
    )
    info['layer_changes'] = layer_changes
    
    # Buscar resultados de episodios
    episodes = re.findall(
        r'EPISODE (\d+) SUMMARY.*?'
        r'Mean reward: ([\d.]+).*?'
        r'Best child this episode: ([\d.]+).*?'
        r'Global best architecture: ([\d.]+)',
        content,
        re.DOTALL
    )
    
    for ep_num, mean_reward, best_child, global_best in episodes:
        info['episodes'].append({
            'episode': int(ep_num),
            'mean_reward': float(mean_reward),
            'best_child': float(best_child),
            'global_best': float(global_best)
        })
    
    # Buscar entrenamientos individuales
    trainings = re.findall(
        r'Child (ep\d+_child\d+) - Training completed:.*?'
        r'Max Val Acc \(last \d+ epochs\) = ([\d.]+), Reward = ([\d.]+)',
        content
    )
    
    for child_id, max_acc, reward in trainings:
        info['architectures'].append({
            'id': child_id,
            'val_acc': float(max_acc),
            'reward': float(reward)
        })
        info['rewards'].append(float(reward))
    
    # Encontrar mejor
    if info['episodes']:
        info['best_reward'] = max(ep['global_best'] for ep in info['episodes'])
    
    return info


def print_summary(info):
    """Imprime resumen de la búsqueda NAS."""
    
    print("=" * 70)
    print("RESUMEN DE BÚSQUEDA NAS")
    print("=" * 70)
    print()
    
    # Configuración
    print("CONFIGURACIÓN:")
    print(f"  • Device: {info.get('device', 'N/A')}")
    print(f"  • Total arquitecturas: {info.get('total_architectures', 'N/A')}")
    print(f"  • Episodes completados: {len(info['episodes'])}")
    print(f"  • Arquitecturas evaluadas: {len(info['architectures'])}")
    print()
    
    # Schedule de capas
    if info['layer_changes']:
        print("SCHEDULE PROGRESIVO DE CAPAS:")
        print(f"  • Inicio: 6 capas")
        for layers, arch_count in info['layer_changes']:
            print(f"  • Después de {arch_count} arquitecturas → {layers} capas")
        print()
    
    # Evolución de rewards
    if info['episodes']:
        print("EVOLUCIÓN DE REWARDS:")
        print(f"  {'Episode':<10} {'Mean Reward':<15} {'Best Child':<15} {'Global Best':<15}")
        print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
        for ep in info['episodes']:
            print(f"  {ep['episode']:<10} "
                  f"{ep['mean_reward']:<15.6f} "
                  f"{ep['best_child']:<15.6f} "
                  f"{ep['global_best']:<15.6f}")
        print()
    
    # Mejores arquitecturas
    if info['architectures']:
        print("TOP 5 ARQUITECTURAS:")
        sorted_archs = sorted(info['architectures'], key=lambda x: x['reward'], reverse=True)[:5]
        print(f"  {'ID':<20} {'Val Acc':<15} {'Reward':<15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15}")
        for arch in sorted_archs:
            print(f"  {arch['id']:<20} {arch['val_acc']:<15.4f} {arch['reward']:<15.6f}")
        print()
    
    # Estadísticas finales
    if info['rewards']:
        print("ESTADÍSTICAS FINALES:")
        print(f"  • Mejor reward encontrado: {max(info['rewards']):.6f}")
        print(f"  • Reward promedio: {sum(info['rewards'])/len(info['rewards']):.6f}")
        print(f"  • Peor reward: {min(info['rewards']):.6f}")
        print()
    
    print("=" * 70)
    print("PROCESO NAS COMPLETADO Y DOCUMENTADO")
    print("=" * 70)


def main():
    """Main function."""
    
    if len(sys.argv) < 2:
        print("Uso: python analyze_nas_logs.py <log_file>")
        print()
        print("Ejemplo:")
        print("  python analyze_nas_logs.py logs/nas_demo/nas_search_20231130_123456.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    print(f"Analizando log: {log_file}")
    print()
    
    info = parse_nas_log(log_file)
    print_summary(info)


if __name__ == "__main__":
    main()
