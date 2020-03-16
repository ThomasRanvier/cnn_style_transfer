import argparse
from stylizer import Stylizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('trained_model_path', type=str, help='path to the trained transfer network to use')
    parser.add_argument('content_image_path', type=str, help='path to the content image to stylize')
    parser.add_argument('--filename', type=str, 
                        default='outputs/images/default.jpg', help='filename of the output image')
    args = parser.parse_args()
    stylizer = Stylizer(args.trained_model_path)
    _ = stylizer.stylize(args.content_image_path, save=True, filename=args.filename)