import argparse
from trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train-dataset-path', type=str, 
                        default='data/coco_data', help='path to the training dataset')
    parser.add_argument('-e', '--epochs', type=int, 
                        default=1, help='number of epochs, default is 1')
    parser.add_argument('-i', '--batch-iterations', type=int, 
                        default=10000, help='number of batch iterations per epoch, default is 10000')
    parser.add_argument('-b', '--batch-size', type=int, 
                        default=4, help='batch size, default is 4')
    parser.add_argument('-lr', '--learning-rate', type=float, 
                        default=1e-3, help='learning rate, default is 1e-3')
    parser.add_argument('-cw', '--content-weight', type=int, 
                        default=1e5, help='content weight, default is 1e5')
    parser.add_argument('-sw', '--style-weight', type=int, 
                        default=1e10, help='style weight, default is 1e10')
    parser.add_argument('-s', '--style-image-path', type=str, 
                        default='data/styles/the_starry_night.jpg', help='style weight, default is the starry night')
    parser.add_argument('-li', '--log-interval', type=int, 
                        default=500, help='log interval (in batches), default is 500')
    parser.add_argument('-si', '--save-interval', type=int, 
                        default=500, help='save interval (in batches), default is 500')
    parser.add_argument('-td', '--trained-models-dir', type=str, 
                        default='outputs/trained_transfer_models', help='path to the trained models directory')
    
    args = parser.parse_args()
    
    ## Create trainer and launch training
    trainer = Trainer(args)
    trainer.train()
    
