from runners.instance_imagenav_runner import Runner
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def main():
    runner = Runner()
    runner.train()

if __name__ == '__main__':
    main()