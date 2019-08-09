import mxnet as mx
import tvm
import nnvm.compiler
import nnvm.testing

inHeight =1024
inWeight = 1057 
device_target = "llvm -mcpu=cortex-a17"

def get_target():
    sym, arg_params, aux_params = mx.model.load_checkpoint("../model/mnet.25/mnet.25", 0)
    target = tvm.target.create(device_target)
    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(sym, arg_params, aux_params)
    print("output list: %s", nnvm_sym.list_output_names())
    return nnvm_sym, nnvm_params, target

def build_store(nnvm_sym, nnvm_params, target):
    image_size = (inHeight, inWeight)
    opt_level=3
    shape_dict = {'data': (1,3,*image_size)}
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(nnvm_sym, target, shape_dict, params=nnvm_params)
    lib.export_library("../model/deploy_lib.so")
    with open("../model/deploy_graph.json", "w") as fo:
        fo.write(graph.json())
    with open("../model/deploy_param.params", "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))

def main():
    nnvm_sym, nnvm_params, target = get_target()
    build_store(nnvm_sym, nnvm_params, target)

if __name__ == '__main__':
    main()
