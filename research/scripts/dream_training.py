"""Native Dream candidate projection and allocator memory reporting."""

from _benchmark_bootstrap import bootstrap

bootstrap()

import torch

from train_qwen_masked_typed import yesno_projection


class CandidateWrapper(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, ids, candidates, positions, noul):
        hidden = self.base.model(input_ids=ids, attention_mask="full", use_cache=False,
                                 return_dict=True).last_hidden_state
        z = yesno_projection(self.base, hidden[0, [p - 1 for p in positions]], candidates)
        return z[0, [1, 0]].unsqueeze(0) if noul else (z[:, 0] - z[:, 1]).unsqueeze(0)


def memory():
    try:
        if not torch.cuda.is_initialized():
            return {}
        free, total = torch.cuda.mem_get_info()
        return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    peak_reserved=torch.cuda.max_memory_reserved(),
                    process_gpu_current_bytes=torch.cuda.memory_allocated(),
                    process_gpu_current_scope="PyTorch allocator; excludes CUDA/NCCL context",
                    device_used_bytes=total - free, device_total_bytes=total)
    except Exception as exc:
        return dict(availability_error=repr(exc))
