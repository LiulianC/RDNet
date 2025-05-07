import torch
from collections import OrderedDict

def analyze_state_dict(path):
    state_dict = torch.load(path, map_location='cpu')
    
    # 统计信息存储
    summary = {
        'model': {'layers': {}, 'total_params': 0, 'total_size': 0},
        'optimizer': {'exists': False},
        'scheduler': {'exists': False}
    }

    def recursive_analyze(d, parent_key='', depth=0):
        if depth > 10:  # 限制递归深度
            return

        for key, value in d.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            
            # 处理张量
            if isinstance(value, torch.Tensor):
                if 'model' in full_key:
                    layer_type = key.split('.')[-2] if '.' in key else 'root'
                    shape = tuple(value.shape)
                    num_params = value.numel()
                    size_mb = value.element_size() * num_params / 1024**2
                    
                    # 汇总层信息
                    if layer_type not in summary['model']['layers']:
                        summary['model']['layers'][layer_type] = {
                            'count': 0,
                            'shapes': set(),
                            'total_params': 0,
                            'total_size': 0
                        }
                    
                    summary['model']['layers'][layer_type]['count'] += 1
                    summary['model']['layers'][layer_type]['shapes'].add(shape)
                    summary['model']['layers'][layer_type]['total_params'] += num_params
                    summary['model']['layers'][layer_type]['total_size'] += size_mb
                    
                    summary['model']['total_params'] += num_params
                    summary['model']['total_size'] += size_mb

            # 识别优化器和调度器
            elif isinstance(value, (dict, OrderedDict)):
                if 'optimizer' in full_key.lower():
                    summary['optimizer']['exists'] = True
                elif 'scheduler' in full_key.lower():
                    summary['scheduler']['exists'] = True
                else:
                    recursive_analyze(value, full_key, depth+1)

    recursive_analyze(state_dict)

    # 打印精简报告
    print("="*50)
    print("模型结构概要")
    print("="*50)
    print(f"-> 总参数量: {summary['model']['total_params']/1e6:.2f}M")
    print(f"-> 总内存占用: {summary['model']['total_size']:.2f}MB\n")
    
    print("主要层类型统计:")
    for layer, info in summary['model']['layers'].items():
        shape_str = ', '.join([f"{s}" for s in sorted(info['shapes'], key=lambda x: sum(x))[:2]]) 
        if len(info['shapes']) > 2:
            shape_str += f" ...(+{len(info['shapes'])-2}种形状)"
        print(f"- {layer:15s} x{info['count']:03d} | 参数范围: {shape_str}")
    
    print("\n训练组件:")
    print(f"优化器状态: {'存在' if summary['optimizer']['exists'] else '无'}")
    print(f"调度器状态: {'存在' if summary['scheduler']['exists'] else '无'}")
    print("="*50)

analyze_state_dict("ytmt_ucs_sirs_latest.pt")