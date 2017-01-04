import os
import sys
import utils


if __name__ == '__main__':
    
    n_argv = len(sys.argv)
    if n_argv != 2:
        raise Exception("Wrong input.")
    else:
        argv = sys.argv
        path = argv[1]
        if not os.path.isfile(path):
            print(Warning("Warning: input '{}' is not a file, skiped.".format(path)))
        else:
            input_path = path
            #output_path = utils.change_extension(input_path, "avi")
            output_path = "output.avi"
            frames = utils.load_tiff(input_path)
            utils.save_avi(frames, output_path)
