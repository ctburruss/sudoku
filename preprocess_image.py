#preprocess puzzle
import cv2
import pytesseract
from PIL import Image
import numpy as np
import math
import sys

def resizeImage(imageFn, maxSize):
	puzzle = cv2.imread(imageFn)
	size = np.shape(puzzle)
	puzzle2 = np.copy(puzzle)
	while np.max(size) > maxSize:
		puzzle2 = cv2.resize(puzzle2, (0,0), fx = 0.8, fy = 0.8)
		size = np.shape(puzzle2)	

	return puzzle2

def findCorners(puzzle):
	gray = cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY)
	ks = 5
	gaussian_kernel = np.ones((ks,ks),np.float32)/pow(ks,2)
	gray_blur = cv2.filter2D(gray, -1, gaussian_kernel)

	#Canny, dilate, and find contours for several thresholds
	edges = cv2.Canny(gray_blur, 50, 150, apertureSize = 3)
	kernel_size = 3
	kernel = np.ones((kernel_size, kernel_size),np.uint8)
	dilation = cv2.dilate(edges,kernel,iterations = 1)
	
	im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	squares = []
	for c in contours:
		epsilon = 0.1*cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, epsilon, True)
		#cv2.waitKey(0)
		if np.shape(approx)[0] == 4:
			squares.append(approx)
	
	maxArea = 0
	gridSquare = squares[0]	
	for s in squares:
		area = cv2.contourArea(s)
		if area >  maxArea:
			maxArea = area
			gridSquare = s
	return gridSquare

def orderPoints(corners):
	cnrs = np.zeros((4,2), dtype = "float32")
	corners = np.array(corners, dtype = "float32")
	s = corners.sum(axis=2)
	diff = np.diff(corners,axis=2)
	cnrs[0] = corners[np.argmin(s)]
	cnrs[2] = corners[np.argmax(s)]
	cnrs[1] = corners[np.argmin(diff)]
	cnrs[3] = corners[np.argmax(diff)]
	return cnrs
	
def projTransform(image, corners):
	corners = orderPoints(corners)
	(tl, tr, br, bl) = corners
	
	widthA = np.sqrt(((br[0]-bl[0]) ** 2) + ((br[1] - bl[1]) **2))
	widthB = np.sqrt(((tr[0]-tl[0]) ** 2) + ((tr[1] - tl[1]) **2))
	maxWidth = max(int(widthA),int(widthB))
	
	heightA = np.sqrt(((tr[0]-br[0]) **2) + ((tr[1] - br[1]) **2))
	heightB = np.sqrt(((tl[0]-bl[0]) **2) + ((tl[1] - bl[1]) **2))
	maxHeight = max(int(heightA),int(heightB))

	dst = np.array([
		[0,0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight -1]], dtype = "float32")
	
	M = cv2.getPerspectiveTransform(corners, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped, M
	
	

def processedGrid(imageFn):
	puzzle = resizeImage(imageFn, 1000)
	corners = findCorners(puzzle)
	transformed_image, M = projTransform(puzzle, corners)
	return transformed_image, corners, M

if __name__=='__main__':
	imageFn = sys.argv[1]
	puzzle,__,__  = processedGrid(imageFn)
	cv2.imshow('puzzle', puzzle)
	cv2.waitKey(0)
	
