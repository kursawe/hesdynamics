import argparse
import os

def make_and_submit_jobscripts(joblist_path):
    '''make and submit a joblist'''
    joblist_directory = os.path.dirname(joblist_path)
    filename_with_extension = os.path.basename(joblist_path)
    filename_without_extension,_ = os.path.splitext(filename_with_extension)

    jobscripts_directory = os.path.join(joblist_directory, filename_without_extension + '_scripts')
   
    if not os.path.exists(jobscripts_directory):
        os.makedirs(jobscripts_directory)

    # number of runs in joblist
    joblist_file = open(joblist_path)
    
    job_index = 0
    for line in joblist_file:
        jobscript_file_name = os.path.join(jobscripts_directory, 'script_'+ str(job_index) + '.sh')
        job_script = open(jobscript_file_name,'w')
        job_script.write('#!/bin/bash')
        job_script.write('\n#$ -S /bin/bash')
        job_script.write('\n#$ -cwd # Job runs in current directory (where you run qsub)')
        job_script.write('\n#$ -V # Job inherits environment (settings from loaded modules etc)')
        job_script.write('\n#$ -pe smp.pe 24 # request intel nodes and find 24 core slot')
        job_script.write('\n#$ -l haswell # Use to specify specific node type')
        job_script.write('\nexport OMP_NUM_THREADS=$NSLOTS # limits jobes to requested resources')
        job_script.write('\nCONDA_PREFIX=/opt/gridware/apps/binapps/anaconda/3/4.2.0')
        job_script.write('\nLD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so ' + line)
        job_script.write('\n\n')
        job_script.write('wait\n')
        job_script.close()
        job_submission_command = 'qsub -l short ' + jobscript_file_name
#         subprocess.Popen(job_sub_command, shell=True)
        job_index +=1

if __name__ == "__main__":
    programtext = "make and submit a jobscript for each line in a file"

    parser = argparse.ArgumentParser(description = programtext, formatter_class=argparse.RawTextHelpFormatter)

    # joblist argument
    parser.add_argument('-j','--joblist', help = 'The path to the file containing the jobs')

    args = parser.parse_args()

    if args.joblist is None:
        print ' *** error: you need to specify a joblist'
        exit()

    dataset_file_name  = os.path.realpath(args.joblist)
    
    make_and_submit_jobscripts(dataset_file_name)
