import numpy as np
from collections import defaultdict

class Sudoku:
	def __init__(self, grid):
		self.grid = grid
		self.tentativeGrid = np.copy(grid)
		self.block = np.zeros(np.shape(grid)) #Block for each element in grid
		self.solved = np.zeros(np.shape(grid))
			
		self.s1, self.s2 = np.shape(grid)
		self.checked = [[[0] for i in range(self.s1)] for j in range(self.s2)]			
		for row in range(0, self.s1):
			for col in range(0,self.s2):
				self.block[row,col] = getBlock(row,col)
		self.possibilities =[[range(1,10) for i in range(self.s1)] for j in range(self.s2)]
		#Eliminate initial possibilities
		self.updatePossibilities()
	
	def updatePossibilities(self):	
		for row in range(0,self.s1):
			for col in range(0,self.s2):
				val = self.grid[row][col]
				if val != 0:
					self.possibilities[row][col] = val
					self.solved[row][col] = 1
				else:				
					r = getRow(row, self.grid)			
					c = getCol(col, self.grid)
					b = getBlockNums(row, col, self)
					elim = r + c + b
					val = self.grid[row][col]
					elim = [i for i in elim if i != val]
					#elim.remove(self.grid[row][col])
					for item in elim:
						if item in self.possibilities[row][col]:
							self.possibilities[row][col].remove(item)					
		
	def updateGrid(self):
		for row in range(0,self.s1):
			for col in range(0, self.s2):
				if np.size(self.possibilities[row][col]) == 1 and self.solved[row][col] !=1:
					self.grid[row][col] = self.possibilities[row][col][0]
					self.solved[row][col] = 1
	
	def solveGrid(self):
		iterations = 0
		while sum(sum(self.solved)) < self.s1 * self.s2:
			self.updatePossibilities()
			self.updateGrid()
			iterations = iterations + 1
			if iterations > 100:
				#print self.grid
				#break
				iterations = self.solveGridBF()
				break
		print 'Game solved in ' + str(iterations) + ' iterations'
		print self.grid
	
	def checkSolution(self,val, row, col):
		r = getRow(row,self.grid)
		c = getCol(col,self.grid)
		b = getBlockNums(row,col, self)
		elim = r + c + b
		if val in elim:
			return 0
		elif val > self.s1:
			return 0
		return 1
	
	def backTrack(self, row, col):
		#Loop through rows, then columns
		if self.solved[row][col] == 0:		
			self.grid[row][col] = 0		
		if col - 1 < 0:
			col = self.s1 - 1
			row = row - 1
					
		else:		
			col = col -1
		if col < 0:
			print 'Broken Puzzle'
		
		#while self.solved[row][col] == 1:
			#row,col = self.backTrack(row,col)

		return row, col
		
	def moveForward(self, row, col):
		if col + 1 >= self.s1:
			col = 0
			row = row+1
		else:
			col = col+1
		
		return row, col
	
	def solveGridBF2(self, grid):
		if sum(sum(grid.solved)) == pow(self.s1,2):
			return grid
		#Find first unassigned
		row = 0
		col = 0
		while grid[row][col] != 0:
			row, col = self.moveForward(row,col)
		for i in range(1,10):
			if self.checkSolution(val, row, col):
				pass
		
	def solveGridBF(self):
		#Use brute force to solve grid
		#Loop through grid, by rows, then columns
		#Put the lowest number not yet checked into the square
		#If this number fails the test, increment it by one
		#If all numbers fail the test, backtrack by one

		row = 0
		col = 0
		
		while self.solved[row][col] == 1:
			row,col = self.moveForward(row,col)

		val = 1
		iterations = 0
		while row < self.s2:
		
			iterations = iterations + 1
			
			while not self.checkSolution(val,row,col):			
				if val < self.s1:
					val = val+1
				else:
					row,col = self.backTrack(row,col)
					while(self.solved[row][col] == 1):				
						row, col = self.backTrack(row,col)
						val = self.grid[row][col]
					val = self.grid[row][col] + 1
				if row == self.s2:
					break
				while(self.solved[row][col] == 1):
					row, col = self.moveForward(row,col)
			self.grid[row][col] = val
			val = 1
			row, col = self.moveForward(row, col)
			
			if row ==9:
				return iterations
			while(self.solved[row][col] == 1):
				row, col = self.moveForward(row,col)
				if row ==9:
					return iterations
			if iterations%1000 == 0:
				print iterations
			if iterations%1000 ==0:
				print self.grid				
				#raw_input('enter to continue')

def getBlock(row,col):
	return row/3*3 +col/3

def getRow(r, puzzle):
	#Return a list of every number in the row
	row = list(puzzle[r,:])
	return row

def getCol(col, puzzle):
	column = list(np.transpose(puzzle[:,col]))
	return column

def getBlockNums(row, col, obj):
	blockNums = []	
	blocks = obj.block
	b = blocks[row][col]
	##Get indices of all other elements in same block
	puzzle = obj.grid
	s1, s2 = np.shape(puzzle)
	for row in range(0,s1):
		for col in range(0,s2):		
			if int(blocks[row][col]) == int(b):		
				blockNums.append(puzzle[row][col])
	return blockNums
