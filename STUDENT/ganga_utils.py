import Ganga, os

def resubmit_failed(job = None):
  job_list = Ganga.GPI.jobs
  if job != None:
    job_list = [job]
  for j in job_list:
    for sj in j.subjobs:
      if sj.status == 'failed': sj.resubmit()

def submit_new(job = None):
  job_list = Ganga.GPI.jobs
  if job != None:
    job_list = [job]
  for j in job_list:
    for sj in j.subjobs:
      if sj.status == 'new': sj.submit()
