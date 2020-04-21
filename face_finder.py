'''Criminal Identification System Using Face Detection and Recognition'''
# KeyWords: Criminal Identification; CCTV; facial recognition; Haar classifier; real-time; Viola-Jones; OpenCV

import collecting_sample as cs
import training_model as tm

print('Scrpit is running now')
criminal_name = str(input("Enter Criminal Name:\t"))

cs.collectingsamples(criminal_name)

