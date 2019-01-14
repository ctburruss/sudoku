from __future__ import division
import cv2
import pytesseract
from PIL import Image
import numpy as np
import math
import preprocess_image as pi
import sys

global DEBUG
DEBUG = 0

def getNumberFromBounds(im, contour, stdThresh):
	[xmin, xmax, ymin, ymax] = contour
	square = im[ymin:ymax, xmin:xmax]
	number = processBB(square, stdThresh)
	return number

def processBB(bb, stdThreshold):
	bb_gray = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
	sy, sx = np.shape(bb_gray)
	mid = bb_gray[int(sy/3):int(2*sy/3), int(sx/3):int(2*sx/3)]
	threshold = 170
	#stdThreshold = 25
	if DEBUG == 1:
		print 'mean = ' + str(np.mean(mid))
		print 'standard deviation = ' + str(np.std(mid))
		cv2.imshow('bb_gray',bb_gray)
		cv2.waitKey(0)
	if np.std(mid) > stdThreshold:
		minAreaThresh = 0.05*np.shape(bb_gray)[0]*np.shape(bb_gray)[1]
		maxAreaThresh = 0.5*np.shape(bb_gray)[0]*np.shape(bb_gray)[1]
		thresh2 = np.mean(mid)
		thresh = cv2.threshold(bb_gray, thresh2, 255, cv2.THRESH_BINARY)[1]	
		im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		mL = 0
		tx, ty = np.shape(thresh)
		if DEBUG == 1:
			cv2.imshow('thresh', thresh)
			cv2.waitKey(0)
		pt = (tx, ty)		
		for c in contours:
			#find longest
			if len(cv2.convexHull(c)) > mL and cv2.contourArea(c) > minAreaThresh and cv2.contourArea(c) < maxAreaThresh:
				mL = len(cv2.convexHull(c))
				cnt = c
		if DEBUG == 1:
			print 'contour Area = ' + str(cv2.contourArea(cnt))
			print 'maxArea = ' + str(maxAreaThresh)
		try:		
			hull = cv2.convexHull(cnt)
		except: #if no contour meets the conditions
			return 0
		x,y,w,h = cv2.boundingRect(hull)
		grow = 5#was 3
		xmin = x-grow
		ymin = y-grow
		xmax = x+w+grow
		ymax = y+h+grow
		
		sy, sx, __ = np.shape(bb)		
		#check edge cases
		if xmax > sx -1:
			xmax = sx -1
		if xmin < 0:
			xmin = 0
		if ymax > sy-1:
			ymax = sy-1
		if ymin < 0:
			ymin = 0 
		bb2 = bb[ymin:ymax,xmin:xmax]

		number = pytesseract.image_to_string(bb2, config='-psm 6')
		
		if DEBUG == 1:
			cv2.drawContours(bb_gray,hull, -1, (0,255,0), 3)			
			cv2.imshow('bb_gray',bb_gray)		
			cv2.imshow('bb2', bb2)		
			cv2.waitKey(0)				
			print number

		if len(number) == 0 or not number.isdigit() or len(number) >1:
			t2 = thresh[ymin:ymax,xmin:xmax]
			number = pytesseract.image_to_string(t2, config='-psm 6')		
		else:
			return number
		return number
	else:
		return 0

def puzzleFromImFn(image, gridSize):
	sy, sx, __ = np.shape(image)
	x = np.linspace(0, sx, gridSize*2 + 1)
	x = [n for i, n in enumerate(x) if i%2 == 1]
	y = np.linspace(0, sy, gridSize*2 + 1)
	y = [n for i, n in enumerate(y) if i%2 == 1]
	return x, y

def getSTDThresh(x_coords, y_coords, image, gridSize):	
	stds = []
	sy, sx, __ = np.shape(image)
	bx = sx/gridSize
	by = sy/gridSize
	for j,x in enumerate(x_coords):
		for i,y in enumerate(y_coords):
			xmin = int(x - bx/2)
			xmax = int(x + bx/2)
			ymin = int(y - by/2)
			ymax = int(y + by/2)
			bounds = [xmin, xmax, ymin, ymax]
			bb = image[ymin:ymax, xmin:xmax]

			bb_gray = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
			sy2, sx2 = np.shape(bb_gray)
			mid = bb_gray[int(sy2/3):int(2*sy2/3), int(sx2/3):int(2*sx2/3)]
			stds.append(np.std(mid))	
	stds.sort()
	smaller = []
	larger = stds
	bestSmaller = []
	bestLarger = stds
	lowestDist = np.inf
	for s in stds:
		smaller.append(s)
		larger.remove(s)
		smean = np.mean(smaller)
		lmean = np.mean(larger)
		dist = 0
		for i in smaller:
			dist = dist + (i - smean)**2
		for j in larger:
			dist = dist + (j - lmean)**2
		if dist < lowestDist:
			lowestDist = dist
			bestSmaller = smaller
			bestLarger = larger
	
	return max(bestSmaller) + (min(bestLarger) - max(bestSmaller))/2
	
def numbersFromGrid(x_coords, y_coords, image, gridSize):
	sy, sx, __ = np.shape(image)	
	bx = sx/gridSize
	by = sy/gridSize
	grid = np.array(np.zeros([gridSize,gridSize]))
	stdThresh = getSTDThresh(x_coords, y_coords, image, gridSize)
	
	for j,x in enumerate(x_coords):
		for i,y in enumerate(y_coords):
			xmin = int(x - bx/2)
			xmax = int(x + bx/2)
			ymin = int(y - by/2)
			ymax = int(y + by/2)
			bounds = [xmin, xmax, ymin, ymax]
			num = getNumberFromBounds(image, bounds, stdThresh)
			grid[i][j] = num
	return grid

def extractGrid(puzzle, gridSize):
	print 'Extracting Puzzle from Image.......'	
	x, y = puzzleFromImFn(puzzle, gridSize)
	grid = numbersFromGrid(x, y, puzzle, gridSize)
	return grid, x, y

if __name__ == '__main__':
	imageFn = sys.argv[1]
	if len(sys.argv) == 3:
		DEBUG = 1
	puzzle, __, __ = pi.processedGrid(imageFn)
	
	if DEBUG == 1:
		cv2.imshow('puzzle',puzzle)
		cv2.waitKey(0)
	grid, _, _ = extractGrid(puzzle, 9)
	print grid
	exit()
	
