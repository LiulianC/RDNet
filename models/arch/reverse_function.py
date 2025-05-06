import torch
from typing import Any, Iterable, List, Tuple, Callable
import torch.distributed as dist

def get_gpu_states(fwd_gpu_devices) -> Tuple[List[int], List[torch.Tensor]]:
    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        with torch.cuda.device(device):
            fwd_gpu_states.append(torch.cuda.get_rng_state())

    return fwd_gpu_states

def get_gpu_device(*args):

    fwd_gpu_devices = list(set(arg.get_device() for arg in args
                               if isinstance(arg, torch.Tensor) and arg.is_cuda))
    return fwd_gpu_devices

def set_device_states(fwd_cpu_state, devices, states) -> None:
    torch.set_rng_state(fwd_cpu_state)
    for device, state in zip(devices, states):
        with torch.cuda.device(device):
            torch.cuda.set_rng_state(state)

def detach_and_grad(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = True
            out.append(x)
        return tuple(out)
    else: 
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)

def get_cpu_and_gpu_states(gpu_devices):
    return torch.get_rng_state(), get_gpu_states(gpu_devices)




# ReverseFunction 是一个自定义的 PyTorch 自动微分函数，用于实现 ​​可逆神经网络的反向传播逻辑​​
# “可逆”并非数学上的严格逆运算，而是指 ​​通过反向传播过程重新计算前向中间变量​​，从而避免在前向时保存所有中间结果，实现​​内存优化​​。
# 其核心思想类似​​梯度检查点（Gradient Checkpointing）​​，但针对残差结构进行了特殊设计。
class ReverseFunction(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, run_functions, alpha, *args):
        l0, l1, l2, l3 = run_functions # 四个阶段的处理函数
        alpha0, alpha1, alpha2, alpha3 = alpha # 各阶段的残差权重
        ctx.run_functions  = run_functions
        ctx.alpha = alpha
        ctx.preserve_rng_state = True

        # 保存随机状态（CPU/GPU RNG、自动混合精度配置）
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}

        assert len(args) == 5
        [x, c0, c1, c2, c3] = args
        if type(c0) == int:
            ctx.first_col = True
        else:
            ctx.first_col = False
        with torch.no_grad():
            gpu_devices = get_gpu_device(*args)
            ctx.gpu_devices = gpu_devices
            ctx.cpu_states_0, ctx.gpu_states_0  = get_cpu_and_gpu_states(gpu_devices)
            c0 = l0(x, c1) + c0*alpha0 # 阶段0  c0_prev是输入的c0，被更新为新值
            ctx.cpu_states_1, ctx.gpu_states_1  = get_cpu_and_gpu_states(gpu_devices)
            c1 = l1(c0, c2) + c1*alpha1 # 阶段1
            ctx.cpu_states_2, ctx.gpu_states_2  = get_cpu_and_gpu_states(gpu_devices)
            c2 = l2(c1, c3) + c2*alpha2 # 阶段3
            ctx.cpu_states_3, ctx.gpu_states_3  = get_cpu_and_gpu_states(gpu_devices)
            c3 = l3(c2, None) + c3*alpha3 # 阶段4
        ctx.save_for_backward(x, c0, c1, c2, c3) # 保存必要张量
        return x, c0, c1 ,c2, c3

    @staticmethod
    def backward(ctx, *grad_outputs):
       
        x, c0, c1, c2, c3 = ctx.saved_tensors
        l0, l1, l2, l3 = ctx.run_functions
        alpha0, alpha1, alpha2, alpha3 = ctx.alpha
        gx_right, g0_right, g1_right, g2_right, g3_right = grad_outputs
        (x, c0, c1, c2, c3) = detach_and_grad((x, c0, c1, c2, c3))

        # pytorch2.3
        # with torch.enable_grad(), \
        #   torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
        #       torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
        #           torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):

        # pytorch2.4
        with torch.enable_grad(), \
            torch.random.fork_rng(devices=ctx.gpu_devices, enabled=ctx.preserve_rng_state), \
                torch.amp.autocast(device_type='cuda', **ctx.gpu_autocast_kwargs),\
                    torch.amp.autocast(device_type='cpu', **ctx.cpu_autocast_kwargs):
            


            
            g3_up = g3_right
            g3_left = g3_up*alpha3 ##shortcut

             # 恢复前向传播的随机状态与自动混合精度配置
            set_device_states(ctx.cpu_states_3, ctx.gpu_devices, ctx.gpu_states_3)                    
            oup3 = l3(c2, None) # 重新计算前向输出
            torch.autograd.backward(oup3, g3_up, retain_graph=True)
            with torch.no_grad():
                c3_left = (1/alpha3)*(c3 - oup3) ## feature reverse # 逆向恢复c2（假设alpha3≠0）
            g2_up = g2_right+ c2.grad
            g2_left = g2_up*alpha2 ##shortcut

            (c3_left,) = detach_and_grad((c3_left,))
            set_device_states(ctx.cpu_states_2, ctx.gpu_devices, ctx.gpu_states_2)          
            oup2 = l2(c1, c3_left)
            torch.autograd.backward(oup2, g2_up, retain_graph=True)
            c3_left.requires_grad = False
            cout3 = c3_left*alpha3 ##alpha3 update
            torch.autograd.backward(cout3, g3_up)
            
            with torch.no_grad():
                c2_left = (1/alpha2)*(c2 - oup2) ## feature reverse
            g3_left = g3_left + c3_left.grad if c3_left.grad is not None else g3_left
            g1_up = g1_right+c1.grad
            g1_left = g1_up*alpha1 ##shortcut

            (c2_left,) = detach_and_grad((c2_left,))
            set_device_states(ctx.cpu_states_1, ctx.gpu_devices, ctx.gpu_states_1)     
            oup1 = l1(c0, c2_left)
            torch.autograd.backward(oup1, g1_up, retain_graph=True)
            c2_left.requires_grad = False
            cout2 = c2_left*alpha2 ##alpha2 update
            torch.autograd.backward(cout2, g2_up)

            with torch.no_grad():
                c1_left = (1/alpha1)*(c1 - oup1) ## feature reverse
            g0_up = g0_right + c0.grad
            g0_left = g0_up*alpha0 ##shortcut
            g2_left = g2_left + c2_left.grad if c2_left.grad is not None else g2_left ## Fusion
            
            (c1_left,) = detach_and_grad((c1_left,))
            set_device_states(ctx.cpu_states_0, ctx.gpu_devices, ctx.gpu_states_0)     
            oup0 = l0(x, c1_left)            
            torch.autograd.backward(oup0, g0_up, retain_graph=True)
            c1_left.requires_grad = False
            cout1 = c1_left*alpha1 ##alpha1 update
            torch.autograd.backward(cout1, g1_up)

            with torch.no_grad():
                c0_left = (1/alpha0)*(c0 - oup0) ## feature reverse
            gx_up = x.grad ## Fusion
            g1_left = g1_left + c1_left.grad if c1_left.grad is not None else g1_left ## Fusion
            c0_left.requires_grad = False
            cout0 = c0_left*alpha0 ##alpha0 update
            torch.autograd.backward(cout0, g0_up)
        
        # 返回梯度（None表示不需要计算run_functions和alpha的梯度）
        if ctx.first_col:
            return None, None, gx_up, None, None, None, None
        else:
            return None, None, gx_up, g0_left, g1_left, g2_left, g3_left


