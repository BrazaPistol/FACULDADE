# pylint: disable=no-member
"""
Flappy Bird com Algoritmo Genético + Fine-tune (Backprop) em Python usando Pygame.
Versão estável e funcional — com janela centralizada, correções completas e melhorias de aprendizado.
"""

import os
import random
import pickle
from pathlib import Path
import numpy as np
import pygame

# -------------------- CONFIGURAÇÕES DE TELA --------------------
os.environ["SDL_VIDEO_CENTERED"] = "1"  # centraliza janela

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730

BASE_DIR = Path(__file__).resolve().parent
IMG_FOLDER = BASE_DIR / "IMG"

# -------------------- INICIALIZAÇÃO --------------------
pygame.init()
pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans", 30)
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird Inteligente - Evolução Correta")

# -------------------- CONFIGURAÇÃO --------------------
CONFIG = {
    "POPULACAO": 150,
    "GENERATIONS": 100,
    "MUTATION_RATE": 0.05,
    "MUTATION_SCALE": 0.01,
    "ELITISM": 5,
    "HIDDEN_NEURONS": 6,
    "USE_BACKPROP": True,
    "BACKPROP_EPOCHS": 5,
    "BACKPROP_LR": 0.1,
    "FPS": 60,
}

DRAW_LINES = False

# -------------------- FUNÇÃO DE IMAGENS --------------------
def load_image(name: str, scale2x: bool = False, size: tuple | None = None):
    path = IMG_FOLDER / name
    if not path.exists():
        surf = pygame.Surface((50, 50), pygame.SRCALPHA)
        surf.fill((255, 0, 255, 128))
        return pygame.transform.scale(surf, size) if size else surf
    img = pygame.image.load(str(path)).convert_alpha()
    if scale2x:
        img = pygame.transform.scale2x(img)
    if size is not None:
        img = pygame.transform.scale(img, size)
    return img

# -------------------- IMAGENS --------------------
pipe_img = load_image("pipe.png", scale2x=True)
bg_img = load_image("bg.png", size=(WIN_WIDTH, 900))
bird_images = [load_image(f"bird{n}.png", scale2x=True) for n in (1, 2, 3)]
base_img = load_image("base.png", scale2x=True)

# -------------------- REDE NEURAL --------------------
def sigmoid(x): return 1 / (1 + np.exp(-x))
def dsigmoid(y): return y * (1 - y)

class RedeNeural:
    def __init__(self, n_inputs=5, n_hidden=6, n_outputs=1):
        self.n_inputs, self.n_hidden, self.n_outputs = n_inputs, n_hidden, n_outputs
        limit1 = np.sqrt(6 / (n_inputs + n_hidden))
        limit2 = np.sqrt(6 / (n_hidden + n_outputs))
        self.w1 = np.random.uniform(-limit1, limit1, (n_hidden, n_inputs))
        self.b1 = np.zeros((n_hidden, 1))
        self.w2 = np.random.uniform(-limit2, limit2, (n_outputs, n_hidden))
        self.b2 = np.zeros((n_outputs, 1))

    def forward(self, x):
        x = np.array(x).reshape(self.n_inputs, 1)
        a1 = sigmoid(self.w1 @ x + self.b1)
        a2 = sigmoid(self.w2 @ a1 + self.b2)
        return a2, a1, x

    def predict(self, x): return float(self.forward(x)[0])

    def backprop_update(self, x, target, lr=0.1):
        out, a1, xcol = self.forward(x)
        y = np.array([[target]])
        error = out - y
        delta2 = error * dsigmoid(out)
        delta1 = (self.w2.T @ delta2) * dsigmoid(a1)
        self.w2 -= lr * (delta2 @ a1.T)
        self.b2 -= lr * delta2
        self.w1 -= lr * (delta1 @ xcol.T)
        self.b1 -= lr * delta1

    def get_flat(self): 
        return np.concatenate([self.w1.flatten(), self.b1.flatten(), self.w2.flatten(), self.b2.flatten()])

    def set_flat(self, flat):
        s1 = self.n_hidden * self.n_inputs
        s2 = self.n_hidden
        s3 = self.n_outputs * self.n_hidden
        s4 = self.n_outputs
        idx = 0
        self.w1 = flat[idx:idx+s1].reshape(self.n_hidden, self.n_inputs); idx += s1
        self.b1 = flat[idx:idx+s2].reshape(self.n_hidden, 1); idx += s2
        self.w2 = flat[idx:idx+s3].reshape(self.n_outputs, self.n_hidden); idx += s3
        self.b2 = flat[idx:idx+s4].reshape(self.n_outputs, 1)

    def copy(self):
        c = RedeNeural(self.n_inputs, self.n_hidden, self.n_outputs)
        c.w1, c.b1, c.w2, c.b2 = self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()
        return c

# -------------------- ALGORITMO GENÉTICO --------------------
class Individuo:
    def __init__(self):
        self.net = RedeNeural()
        self.fitness = 0.0
    def copy(self):
        i = Individuo()
        i.net = self.net.copy()
        i.fitness = self.fitness
        return i

class AlgoritmoGenetico:
    def __init__(self):
        self.population = [Individuo() for _ in range(CONFIG["POPULACAO"])]

    def evaluate(self, fits):
        for ind, fit in zip(self.population, fits): ind.fitness = fit

    def select(self):
        a, b = random.sample(self.population, 2)
        return a if a.fitness > b.fitness else b

    def crossover(self, p1, p2):
        f1, f2 = p1.net.get_flat(), p2.net.get_flat()
        mask = np.random.rand(len(f1)) > 0.5
        child = Individuo()
        child.net.set_flat(np.where(mask, f1, f2))
        return child

    def mutate(self, ind):
        flat = ind.net.get_flat()
        mask = np.random.rand(len(flat)) < CONFIG["MUTATION_RATE"]
        flat[mask] += np.random.normal(0, CONFIG["MUTATION_SCALE"], mask.sum())
        ind.net.set_flat(flat)

    def next_generation(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        new_pop = [self.population[i].copy() for i in range(CONFIG["ELITISM"])]
        while len(new_pop) < CONFIG["POPULACAO"]:
            c = self.crossover(self.select(), self.select())
            self.mutate(c)
            new_pop.append(c)
        self.population = new_pop

# -------------------- ENTIDADES --------------------
class Bird:
    IMGS = bird_images
    MAX_ROT = 25
    ROT_VEL = 20
    ANIM_TIME = 5

    def __init__(self, x, y):
        self.x, self.y, self.vel, self.tilt = x, y, 0, 0
        self.tick_count, self.height, self.img_count = 0, y, 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        d = self.vel * self.tick_count + 1.5 * (self.tick_count ** 2)
        d = max(min(d, 16), -16)
        self.y += d
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROT:
                self.tilt = self.MAX_ROT
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1
        frame = (self.img_count // self.ANIM_TIME) % 4
        self.img = self.IMGS[frame % 3]
        blit_rotate_center(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self): return pygame.mask.from_surface(self.img)

class Pipe:
    GAP, VEL = 180, 5
    def __init__(self, x):
        self.x = x
        self.height = random.randrange(50, 450)
        self.top = self.height - pipe_img.get_height()
        self.bottom = self.height + self.GAP
        self.pipe_top = pygame.transform.flip(pipe_img, False, True)
        self.pipe_bottom = pipe_img
        self.passed_by = set()
    def move(self): self.x -= self.VEL
    def draw(self, win):
        win.blit(self.pipe_top, (self.x, self.top))
        win.blit(self.pipe_bottom, (self.x, self.bottom))
    def collide(self, bird):
        bmask = bird.get_mask()
        tmask = pygame.mask.from_surface(self.pipe_top)
        bmask2 = pygame.mask.from_surface(self.pipe_bottom)
        t_offset = (self.x - bird.x, self.top - round(bird.y))
        b_offset = (self.x - bird.x, self.bottom - round(bird.y))
        return bmask.overlap(tmask, t_offset) or bmask.overlap(bmask2, b_offset)

class Base:
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img
    def __init__(self, y):
        self.y = y
        self.x1, self.x2 = 0, self.WIDTH
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0: self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0: self.x2 = self.x1 + self.WIDTH
    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

# -------------------- FUNÇÕES AUXILIARES --------------------
def blit_rotate_center(surf, image, topleft, angle):
    rotated = pygame.transform.rotate(image, angle)
    rect = rotated.get_rect(center=image.get_rect(topleft=topleft).center)
    surf.blit(rotated, rect.topleft)

def normalize_inputs(bird, pipe):
    return [
        bird.y / WIN_HEIGHT,
        (pipe.height - bird.y) / WIN_HEIGHT,
        (pipe.bottom - bird.y) / WIN_HEIGHT,
        bird.vel / 20.0,
        (pipe.x - bird.x) / WIN_WIDTH,
    ]

# -------------------- LOOP DE JOGO (AVALIAÇÃO) --------------------
def eval_population(alg, display=True, max_frames=6000):
    nets = [ind.net for ind in alg.population]
    birds = [Bird(230, 350) for _ in alg.population]
    base, pipes = Base(FLOOR), [Pipe(600)]
    alive = [True] * len(birds)
    fitness = [0] * len(birds)
    score, frame = 0, 0
    clock = pygame.time.Clock()

    while frame < max_frames and any(alive):
        clock.tick(CONFIG["FPS"])
        frame += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); raise SystemExit

        # Atualiza lógica
        for i, bird in enumerate(birds):
            if not alive[i]: continue
            pipe = next((p for p in pipes if p.x + p.pipe_top.get_width() > bird.x), pipes[0])
            out = nets[i].predict(normalize_inputs(bird, pipe))
            if out > 0.5: bird.jump()
            bird.move()
            fitness[i] += 0.05

        add_pipe, rem = False, []
        for p in pipes:
            p.move()
            for i, b in enumerate(birds):
                if not alive[i]: continue
                if p.collide(b):
                    alive[i] = False; fitness[i] -= 1
                elif i not in p.passed_by and b.x > p.x + p.pipe_top.get_width():
                    p.passed_by.add(i); fitness[i] += 5
                    add_pipe = True
            if p.x + p.pipe_top.get_width() < 0: rem.append(p)
        if add_pipe:
            pipes.append(Pipe(WIN_WIDTH)); score += 1
        for r in rem: pipes.remove(r)

        # chão/teto
        for i, b in enumerate(birds):
            if not alive[i]: continue
            if b.y + b.img.get_height() >= FLOOR or b.y < -50:
                alive[i] = False

        if display:
            WIN.blit(bg_img, (0, 0))
            for p in pipes: p.draw(WIN)
            base.draw(WIN)
            for b in [birds[i] for i in range(len(birds)) if alive[i]]: b.draw(WIN)
            WIN.blit(STAT_FONT.render(f"Score: {score}", 1, (255,255,255)), (450, 10))
            WIN.blit(STAT_FONT.render(f"Alive: {sum(alive)}", 1, (255,255,255)), (10, 10))
            pygame.display.update()

    return fitness

# -------------------- FINE-TUNE --------------------
def fine_tune_elites(alg, k=None):
    if k is None: k = CONFIG["ELITISM"]
    alg.population.sort(key=lambda x: x.fitness, reverse=True)
    for elite in alg.population[:k]:
        for _ in range(3):
            samples = []
            b, pipes = Bird(230,350), [Pipe(600)]
            for _ in range(200):
                p = next((px for px in pipes if px.x + px.pipe_top.get_width() > b.x), pipes[0])
                x = normalize_inputs(b, p)
                y = 1.0 if elite.net.predict(x) > 0.5 else 0.0
                samples.append((x, y))
                if y: b.jump()
                b.move()
                for px in pipes:
                    px.move()
                    if px.collide(b): b.y = FLOOR
                if b.y >= FLOOR: break
            target = 1 if b.y < FLOOR else 0
            for _ in range(CONFIG["BACKPROP_EPOCHS"]):
                for s in samples:
                    elite.net.backprop_update(s[0], target, lr=CONFIG["BACKPROP_LR"])

# -------------------- EVOLUÇÃO PRINCIPAL --------------------
def run_evolution():
    alg = AlgoritmoGenetico()
    if Path("best_net.pkl").exists():
        with open("best_net.pkl", "rb") as f:
            flat = pickle.load(f)
            alg.population[0].net.set_flat(flat)
        print("best_net.pkl carregado.")

    for gen in range(1, CONFIG["GENERATIONS"]+1):
        print(f"\n=== Geração {gen} ===")
        fits = eval_population(alg, display=True)
        alg.evaluate(fits)
        print(f"Melhor: {max(fits):.2f} | Média: {np.mean(fits):.2f}")

        if CONFIG["USE_BACKPROP"]:
            fine_tune_elites(alg)

        alg.next_generation()
        alg.population.sort(key=lambda x: x.fitness, reverse=True)
        best = alg.population[0].net
        with open("best_net.pkl", "wb") as f: pickle.dump(best.get_flat(), f)
        print("Checkpoint salvo.")

    print("Treinamento concluído!")

# -------------------- MAIN --------------------
if __name__ == "__main__":
    run_evolution()
    pygame.quit()
