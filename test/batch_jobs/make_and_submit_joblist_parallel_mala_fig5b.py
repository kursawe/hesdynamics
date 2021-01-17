#!python
import argparse
import os
import subprocess
import time
import make_dataset_job_list_fig5b

def make_and_submit_jobscripts(joblist_path):
    '''make and submit a joblist'''
    joblist_directory = os.path.dirname(joblist_path)
    filename_with_extension = os.path.basename(joblist_path)
    filename_without_extension,_ = os.path.splitext(filename_with_extension)

    jobscripts_directory = os.path.join(joblist_directory, filename_without_extension + '_scripts')

    if not os.path.exists(jobscripts_directory):
        os.makedirs(jobscripts_directory)

    # number of runs in joblist
    job_index = 0
    for job_file in os.listdir(joblist_path):
        print(1)
        job_file = open(os.path.join(joblist_path,job_file))
        for line in job_file:
            jobscript_file_name = os.path.join(jobscripts_directory, 'script_'+ line[line.find('ps'):line.find('.npy')] + '.sh')
            job_script = open(jobscript_file_name,'w')
            job_script.write('#!/bin/bash --login')
            job_script.write('\n#$ -cwd # Job runs in current directory (where you run qsub)')
            job_script.write('\n#$ -V # Job inherits environment (settings from loaded modules etc)')
            job_script.write('\n#$ -pe smp.pe 8 # request intel nodes and find 8 core slot')
            job_script.write('\n#$ -M joshua.burton@manchester.ac.uk')
            job_script.write('\n#$ -m e # send email to above address when job ends')
            job_script.write('\nexport OMP_NUM_THREADS=1 # limits jobs to requested resources')
            job_script.write('\ncd ~/scratch/$USER/hesdynamics/test/')
            job_script.write('\n' + line)
            job_script.write('\n\n')
            job_script.write('wait\n')
            job_script.close()
            job_submission_command = 'qsub ' + jobscript_file_name
            print('submitting job with this command:')
            print(job_submission_command)
            subprocess.Popen(job_submission_command, shell=True)
            time.sleep(1)
        job_index +=1

if __name__ == "__main__":
    # make joblists
    saving_path = make_dataset_job_list_fig5b.main()
    make_and_submit_jobscripts(saving_path)
