import pygame
import random
import os
import neat
pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 700

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird1.png'))),
			pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird2.png'))),
			pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird3.png')))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'pipe.png')))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'base.png')))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bg.png')))
STAT_FONT = pygame.font.SysFont("inconsolata", 40)

NUMBER_OF_GENERATIONZ = 1000

GEN = 0

class Bird(object):

	IMGS = BIRD_IMGS
	MAX_ROTATION = 25
	ROT_VEL = 20
	ANIMATION_TIME = 5

	def __init__(self, x, y):

		self.x = x
		self.y = y
		self.tilt = 0
		self.nTicks = 0
		self.vel = 0
		self.height = self.y
		self.nImgs = 0
		self.img = self.IMGS[0]

	def jump(self):

		self.vel = -10.5
		self.nTicks = 0
		self.height = self.y

	def move(self):

		self.nTicks += 1

		d = self.vel*self.nTicks + 1.5*self.nTicks**2

		if d >= 16:
			d = 16
		if d < 0:
			d -= 2

		self.y += d

		if d < 0 or (self.y < self.height + 50):
			if self.tilt < self.MAX_ROTATION:
				self.tilt = self.MAX_ROTATION
		else:
			if self.tilt > -90:
				self.tilt -= self.ROT_VEL

	def draw(self, window):

		self.nImgs += 1

		if self.nImgs < self.ANIMATION_TIME:
			self.img = self.IMGS[0]
		elif self.nImgs < self.ANIMATION_TIME*2:
			self.img = self.IMGS[1]
		elif self.nImgs < self.ANIMATION_TIME*3:
			self.img = self.IMGS[2]
		elif self.nImgs < self.ANIMATION_TIME*4:
			self.img = self.IMGS[1]
		elif self.nImgs < self.ANIMATION_TIME*4 + 1:
			self.img = self.IMGS[0]
			self.nImgs = 0

		if self.tilt <= -80:

			self.img = self.IMGS[1]
			self.nImgs = self.ANIMATION_TIME*2

		rotatedImg = pygame.transform.rotate(self.img, self.tilt)
		newRect = rotatedImg.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
		window.blit(rotatedImg, newRect)

	def getMask(self):

		return pygame.mask.from_surface(self.img)

class Pipe(object):

	GAP = 200
	VEL = 5

	def __init__(self, x):

		self.x = x
		self.height = 0

		self.top = 0
		self.bottom = 0
		self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
		self.PIPE_BOTTOM = PIPE_IMG

		self.passed = False
		self.setHeight()

	def setHeight(self):

		self.height = random.randint(50, 450)
		self.top = self.height - self.PIPE_TOP.get_height()
		self.bottom = self.height + self.GAP

	def move(self):

		self.x -= self.VEL

	def draw(self, window):

		window.blit(self.PIPE_TOP, (self.x, self.top))
		window.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

	def collide(self, bird):

		birdMask = bird.getMask()
		topMask = pygame.mask.from_surface(self.PIPE_TOP)
		bottomMask = pygame.mask.from_surface(self.PIPE_BOTTOM)

		topOffset = (self.x - bird.x, self.top - round(bird.y))
		bottomOffset = (self.x - bird.x, self.bottom - round(bird.y))

		bPoint = birdMask.overlap(bottomMask, bottomOffset)
		tPoint = birdMask.overlap(topMask, topOffset)

		if tPoint or bPoint:
			return True

		return False

class Base(object):

	VEL = 5
	WIDTH = BASE_IMG.get_width()
	IMG = BASE_IMG

	def __init__(self, y):

		self.y = y
		self.x1 = 0
		self.x2 = self.WIDTH

	def move(self):

		self.x1 -= self.VEL
		self.x2 -= self.VEL

		if self.x1 + self.WIDTH < 0:
			self.x1 = self.x2 + self.WIDTH

		if self.x2 + self.WIDTH < 0:
			self.x2 = self.x1 + self.WIDTH

	def draw(self, window):

		window.blit(self.IMG, (self.x1, self.y))
		window.blit(self.IMG, (self.x2, self.y))


def makeWindow(window, birds, pipes, base, score, gen):

	window.blit(BG_IMG, (0, 0))

	text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
	window.blit(text, (WIN_WIDTH-10-text.get_width(), 10))

	text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
	window.blit(text, (10, 10))

	base.draw(window)
	for pipe in pipes:
		pipe.draw(window)
	for bird in birds:
		bird.draw(window)

	pygame.display.flip()

def main(genomes, config):

	global GEN
	GEN += 1

	pygame.init()
	gameWindow = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
	running = True
	clock = pygame.time.Clock()

	nets = []
	ge = []
	birds = []

	for _, genome in genomes:
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		nets.append(net)
		birds.append(Bird(230, 350))
		genome.fitness = 0
		ge.append(genome)

	pipes = [Pipe(600)]
	base = Base(650)
	score = 0

	while running:

		clock.tick(30)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
				pygame.quit()
				quit()

		pipeInd = 0
		if len(birds) > 0:
			if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
				pipeInd = 1
		else:
			running = False
			break

		for x, bird in enumerate(birds):
			bird.move()
			ge[x].fitness += 0.1

			output = nets[x].activate((bird.y, abs(bird.y-pipes[pipeInd].height), abs(bird.y-pipes[pipeInd].bottom)))

			if output[0] > 0.5:
				bird.jump()

		addPipe = False
		rem = []

		for pipe in pipes:
			for x, bird in enumerate(birds):
				if pipe.collide(bird):
					ge[x].fitness -= 1
					birds.pop(x)
					nets.pop(x)
					ge.pop(x)

				if not pipe.passed and pipe.x < bird.x:
					pipe.passed = True
					addPipe = True

			if pipe.x + pipe.PIPE_TOP.get_width() < 0:
				rem.append(pipe)

			pipe.move()

		if addPipe:
			score += 1
			for g in ge:
				g.fitness += 5
			pipes.append(Pipe(600))

		for r in rem:
			pipes.remove(r)

		for x, bird in enumerate(birds):
			if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
				birds.pop(x)
				nets.pop(x)
				ge.pop(x)

		base.move()
		makeWindow(gameWindow, birds, pipes, base, score, GEN)

def run(configPath):

	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, configPath)

	p = neat.Population(config)

	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	winner = p.run(main, NUMBER_OF_GENERATIONZ)

if __name__ == '__main__':
	localDir = os.path.dirname(__file__)
	configPath = os.path.join(localDir, 'config.txt')
	run(configPath)
