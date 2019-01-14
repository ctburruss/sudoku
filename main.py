#!/usr/bin/env python
import time
import sys
import cv2
import postprocess_image as pp
import os

def testAllPuzzles(directory):
	puzzleFns = os.listdir(directory)
	for fn in puzzleFns:
		puzzleFn = os.path.join(directory, fn)
		print 'Solving puzzle in image ' + puzzleFn
		pp.solveAndDisplay(puzzleFn)	
		cv2.destroyAllWindows()

if __name__ == '__main__':	
	if sys.argv[1] == '-testAll':
		testDir = sys.argv[2]		
		testAllPuzzles(testDir)
	else:
		imageFn = sys.argv[1]
		start_time = time.time()
		pp.solveAndDisplay(imageFn)
		print 'total time = ' + str(time.time() - start_time)




		


