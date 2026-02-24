from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs/train")
global_step = 0

for step in range(100):
    if step < 50:
        writer.add_scalar("loss", 0.1 * step, global_step)       # , step
    else:
        writer.add_scalar("loss", 0.1 * (100-step), global_step)
    global_step += 1

writer.close()

'''
run tensorboard:
tensorboard --logdir=logs


visit :
http://localhost:6006
'''