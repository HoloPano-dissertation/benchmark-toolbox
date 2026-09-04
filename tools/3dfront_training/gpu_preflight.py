
import argparse
import json
import torch


def check_gpu(require_detectron=False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU allocation is required")
    capability = torch.cuda.get_device_capability()
    arch = "sm_%d%d" % capability
    supported = torch.cuda.get_arch_list()
    if arch not in supported:
        raise RuntimeError("GPU %s (%s) is not compiled into this PyTorch build (%s). "
                           "Use a supported allocation or a separate compatible environment."
                           % (torch.cuda.get_device_name(), arch, supported))
    layer = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
    x = torch.randn(2, 3, 32, 64, device="cuda", requires_grad=True)
    layer(x).square().mean().backward()
    if require_detectron:
        from detectron2.layers import nms_rotated
        boxes = torch.tensor([[10., 10., 5., 5., 0.]], device="cuda")
        scores = torch.ones(1, device="cuda")
        assert len(nms_rotated(boxes, scores, 0.5)) == 1
    torch.cuda.synchronize()
    result = {"gpu": torch.cuda.get_device_name(), "capability": capability,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "convolution_backward": "passed", "detectron_cuda": require_detectron}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detectron", action="store_true")
    check_gpu(parser.parse_args().detectron)
