'''Criminal Identification System Using Face Detection and Recognition'''
# KeyWords: Criminal Identification; CCTV; facial recognition; Haar classifier; real-time; Viola-Jones; OpenCV

import collecting_sample as cs




print('Scrpit is running now')
while(True):
    print('What do you want?\n'
          '1. Feed Data with new Face\n'
          "2. Recognize Image\n"
          "3. exit")
    userinput = int(input("Enter your choice-->\n"))

    if userinput == 1:
        criminal_name = str(input("Enter Criminal Name:\t"))
        criminal_id = int(input("Enter Criminal ID\t"))

        cs.collectingsamples(criminal_name, criminal_id)
    elif userinput == 2:
        import training_model as tm
        tm.model_train();

    else:
        exit(0)



# import training_model as tm
# tm.model_train()
