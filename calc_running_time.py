import torch
import utils
import numpy as np


def calc_time(model):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    running_times = {}
    for image_size in [256,512]:
        x = torch.randn(1,1,image_size,image_size).cuda()
        model.eval()

        # warmup
        for _ in range(10): y_est = model(x)
        
        times = []
        for _ in range(25):
            with torch.no_grad():
                start.record()
                y_est = model(x)
                end.record()

                # Waits for everything to finish running
                torch.cuda.synchronize()

                times.append(start.elapsed_time(end))
        
        running_times[image_size] = np.mean(times)
    
    return running_times


if __name__ == '__main__':
    
    model = utils.get_model('dncnn')
    m = model(num_neurons=1024,num_channels=1).cuda()
    print(calc_time(m))

    model = utils.get_model('superonn')
    for q in [1,3,5,7]:
        print('='*64)
        print('q=',q)
        print('='*64)
        for num_neurons in [256,512]:
            for max_shift in [[0,0,0],[0,5,0]]:
                print(q,num_neurons,max_shift)
                m = model(num_neurons=num_neurons,q=q,max_shifts=max_shift,num_channels=1).cuda()
                print(calc_time(m))

