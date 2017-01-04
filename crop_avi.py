import os
import sys
import utils


if __name__ == '__main__':
    
    n_argv = len(sys.argv)
    if n_argv != 6:
        raise Exception("Wrong input.")
    else:
        argv = sys.argv
        path = argv[1]
        left_offset = int(argv[2])
        right_offset = int(argv[3])
        up_offset = int(argv[4])
        down_offset = int(argv[5])
        if not os.path.isfile(path):
            print(Warning("Warning: input '{}' is not a file, skiped.".format(path)))
        else:
            input_path = path
            offsets = (
                left_offset,
                right_offset,
                up_offset,
                down_offset
            )
            output_path = "output_crop.avi"
            frames = utils.load_avi(input_path)
            frames = utils.crop(frames, offsets)
            utils.save_avi(frames, output_path)
