import torch
from torch.export import export
import pytest

import tvm
from tvm import relax
import tvm.testing
from tvm.relax.frontend.torch import from_exported_program


def verify_model(torch_model, example_args, example_kwargs={}, target: str = "llvm", dev=tvm.cpu()):
    # PyTorch
    exported_program = export(torch_model, args=example_args, kwargs=example_kwargs)
    torch_output: torch.Tensor = exported_program.module()(*example_args)

    # Relax
    mod = from_exported_program(exported_program)
    mod = tvm.relax.transform.DecomposeOpsForInference()(mod)
    exe = relax.build(mod, target=target)
    vm = relax.VirtualMachine(exe, dev)
    tvm_args = [tvm.nd.from_dlpack(x.contiguous()) for x in example_args]
    tvm_output = vm["main"](*tvm_args)

    if isinstance(torch_output, tuple):
        expected = torch.stack(torch_output)
        actual = torch.stack([torch.from_numpy(x.numpy()) for x in tvm_output])
    else:
        expected = torch_output
        actual = torch.from_numpy(tvm_output[0].numpy())

    torch.testing.assert_close(
        actual.shape,
        expected.shape,
        msg=f"expected: {expected.shape}, actual: {actual.shape}",
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
        equal_nan=True,
    )


def verify_torchvision_model(model_name):
    from tvm.contrib.download import download_testdata
    from torchvision.models import get_model, get_model_weights
    from torchvision.io import read_image

    # prepare sample image
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_name = "cat.png"
    img_path = download_testdata(img_url, img_name, module="data")
    image_tensor = read_image(img_path)

    model = get_model(model_name, weights="DEFAULT").eval()
    weights = get_model_weights(model_name).DEFAULT
    transforms = weights.transforms()

    batch = transforms(image_tensor).unsqueeze(0)
    example_args = (batch,)
    verify_model(model, example_args)


def test_e2e_alexnet():
    verify_torchvision_model("alexnet")


def test_e2e_convnext_tiny():
    verify_torchvision_model("convnext_tiny")


def test_e2e_densenet121():
    verify_torchvision_model("densenet121")


def test_e2e_efficientnet_b0():
    verify_torchvision_model("efficientnet_b0")


def test_e2e_efficientnet_v2_s():
    verify_torchvision_model("efficientnet_v2_s")


def test_e2e_inception_v3():
    verify_torchvision_model("inception_v3")


def test_e2e_maxvit_t():
    verify_torchvision_model("maxvit_t")


def test_e2e_mnasnet0_5():
    verify_torchvision_model("mnasnet0_5")


def test_e2e_mobilenet_v2():
    verify_torchvision_model("mobilenet_v2")


def test_e2e_mobilenet_v3_small():
    verify_torchvision_model("mobilenet_v3_small")


def test_e2e_regnet_x_400mf():
    verify_torchvision_model("regnet_x_400mf")


def test_e2e_resnet18():
    verify_torchvision_model("resnet18")


def test_e2e_resnext50_32x4d():
    verify_torchvision_model("resnext50_32x4d")


def test_e2e_shufflenet_v2_x0_5():
    verify_torchvision_model("shufflenet_v2_x0_5")


def test_e2e_squeezenet1_0():
    verify_torchvision_model("squeezenet1_0")


def test_e2e_swin_t():
    verify_torchvision_model("swin_t")


def test_e2e_swin_v2_t():
    verify_torchvision_model("swin_v2_t")


def test_e2e_vgg11():
    verify_torchvision_model("vgg11")


def test_e2e_vgg11_bn():
    verify_torchvision_model("vgg11_bn")


def test_e2e_vit_b_32():
    verify_torchvision_model("vit_b_32")


def test_e2e_wide_resnet50_2():
    verify_torchvision_model("wide_resnet50_2")


if __name__ == "__main__":
    tvm.testing.main()
