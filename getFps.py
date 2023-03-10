import torch
import time
from model.dev_ctr_sa1_da_aff import Model


def measure_inference_speed(model, data, max_iter=150, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            fps_str = f'Overall fps: {fps:.1f} img / s, times per image: {1000 / fps:.1f} ms / img'
            print(fps_str, flush=True)
            break
    return fps, fps_str
model = Model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = model.to(device)
# bs = 1
# print('bs: ', bs)
# print(net)

# bs_list = [1,16]
bs_list = [1,16,32,48,64,128,256]

fps_str_list = []
for bs in bs_list:
  print('bs: ', bs)
  data = torch.rand((bs, 3, 64, 25, 2)).to(device)
  fps, fps_str = measure_inference_speed(net, (data,))
  fps_str_list.append('bs: '.format(bs) + fps_str)

print('name: ', model.__class__)
for fps_str in fps_str_list:
  print(fps_str)



# bs 1
# dev_ctr_sa1_aff Overall fps: 38.4 img / s, times per image: 26.1 ms / img
# ctr Overall fps: 48.0 img / s, times per image: 20.8 ms / img  推理一张26.1


# bs 64
# dev_ctr_sa1_aff Overall fps: 8.1 img / s, times per image: 123.6 ms / img
# ctr Overall fps: 10.2 img / s, times per image: 98.1 ms / img  # 推理664张,98.1
# sectr Overall fps: 7.1 img / s, times per image: 140.4 ms / img
'''

name:  <class 'model.ctrgcn.Model'>
Overall fps: 59.2 img / s, times per image: 16.9 ms / img
Overall fps: 18.1 img / s, times per image: 55.3 ms / img
Overall fps: 9.0 img / s, times per image: 110.7 ms / img
Overall fps: 6.0 img / s, times per image: 166.1 ms / img
Overall fps: 4.5 img / s, times per image: 224.4 ms / img
Overall fps: 2.2 img / s, times per image: 450.1 ms / img
Overall fps: 1.1 img / s, times per image: 909.4 ms / img

name:  <class 'model.dev_ctr_sa1_da_aff.Model'>
bs: Overall fps: 40.8 img / s, times per image: 24.5 ms / img
bs: Overall fps: 13.9 img / s, times per image: 71.8 ms / img
bs: Overall fps: 7.0 img / s, times per image: 142.7 ms / img
bs: Overall fps: 4.7 img / s, times per image: 213.7 ms / img
bs: Overall fps: 3.5 img / s, times per image: 288.7 ms / img
bs: Overall fps: 1.7 img / s, times per image: 578.5 ms / img
bs: Overall fps: 0.9 img / s, times per image: 1164.6 ms / img


name:  <class 'model.infogcn.InfoGCN'>
bs: Overall fps: 83.2 img / s, times per image: 12.0 ms / img
bs: Overall fps: 21.5 img / s, times per image: 46.5 ms / img
bs: Overall fps: 10.8 img / s, times per image: 92.8 ms / img
bs: Overall fps: 7.2 img / s, times per image: 139.7 ms / img
bs: Overall fps: 5.3 img / s, times per image: 189.5 ms / img
bs: Overall fps: 2.6 img / s, times per image: 378.3 ms / img
bs: Overall fps: 1.3 img / s, times per image: 768.1 ms / img



bs1

Overall fps: 58.2 img / s, times per image: 17.2 ms / img
name:  <class 'model.ctrgcn.Model'>  bs:  1

r9000p
dev_ctr_sa1_aff  64
Overall fps: 3.6 img / s, times per image: 274.4 ms / img

ctr  64
Overall fps: 4.5 img / s, times per image: 220.4 ms / img
'''
