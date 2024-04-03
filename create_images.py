from typing import List
import sys
sys.path.append("ecgimagegenerator")
import ecgimagegenerator.gen_ecg_images_from_data_batch
import argparse
import prepare_ptbxl_data
import add_image_filenames
import os
import



def generate_images(input_dirs:List[str], output_dir:str):
    for input_dir in input_dirs:
        args = argparse.ArgumentParser(description='Bad Solution')
        args.input_folder = f"{input_dir}"
        args.database_file = "/home/tdege/DeTECRohr/PhysioNet2024/ptb-xl/ptbxl_database.csv"
        args.statements_file = "/home/tdege/DeTECRohr/PhysioNet2024/ptb-xl/scp_statements.csv"
        args.output_folder = f"{output_dir}"
        prepare_ptbxl_data.run(args)


    arg_raw = ["--input_directory", f"{output_dir}", "--output_directory", f"{output_dir}"]
    ecgimagegenerator.gen_ecg_images_from_data_batch.main(arg_raw)

    args = argparse.ArgumentParser(description='Bad Solution')
    args.input_folder = f"{output_dir}"
    args.output_folder = f"{output_dir}"
    add_image_filenames.run(args)




if __name__ == '__main__':
    directory = "/home/tdege/DeTECRohr/PhysioNet2024/ptb-xl/records100"
    input_dirs = [f.path for f in os.scandir(directory) if f.is_dir()]
    output_dir = "/home/tdege/DeTECRohr/PhysioNet2024/ptb-xl/test_skr"
    generate_images(input_dirs[:19], output_dir)