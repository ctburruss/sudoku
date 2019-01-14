from __future__ import division
import cv2
from PIL import Image
import numpy as np
import preprocess_image as pi
import sys
import sudoku_cv as sc
import sudoku_utils as su

def overlaySolution(image, initial_state, solvedPuzzle, x_coords, y_coords):
	font = cv2.FONT_HERSHEY_SIMPLEX
	sy, sx, __ = np.shape(image)
	dx = sx/np.shape(solvedPuzzle)[0]
	dy = sy/np.shape(solvedPuzzle)[0]
	for j, x in enumerate(x_coords):
		for i, y in enumerate(y_coords):
			if int(initial_state[i][j]) != 1:
				cv2.putText(image,str(int(solvedPuzzle[i][j])),(int(x)-10,int(y)+10), font, 1,(50,50,50),3,cv2.LINE_AA)
	return image

def overlaySolutionSkewed(originalCorners, originalImage, newImage, M):
	rows, cols, __ = np.shape(originalImage)
	_,M = cv2.invert(M)

	unwarped = cv2.warpPerspective(newImage, M, (cols, rows),)
	cv2.fillPoly(originalImage, pts =[originalCorners], color=(0,0,0))
	final2 = cv2.add(unwarped, originalImage)
	return final2

def beforeAfter(originalImage, finalImage):
	both = np.concatenate((originalImage, finalImage), axis=1)
	cv2.imshow('Solved Sudoku', both)
	cv2.waitKey(0)
	
def solveAndDisplay(imageFn):
	image1 = pi.resizeImage(imageFn,1000)
	puzzle_image, corners, M = pi.processedGrid(imageFn)
	puzzle, x, y = sc.extractGrid(puzzle_image, 9)
	game = su.Sudoku(puzzle)
	print game.grid
	initial_state = np.copy(game.solved)
	game.solveGrid()
	newImage = overlaySolution(puzzle_image, initial_state, game.grid, x, y)
	finalImage = overlaySolutionSkewed(corners, image1, newImage, M)
	beforeAfter(pi.resizeImage(imageFn,1000), finalImage)

if __name__ == '__main__':
	DEBUG = 0
	imageFn = sys.argv[1]
	solveAndDisplay(imageFn)
	
	
