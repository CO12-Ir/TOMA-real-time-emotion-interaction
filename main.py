from threading import Thread
import pygame
import record2

def run_pygame():
    global toma_face
    HEAD_RECT = pygame.Rect(150, 0, 200, 200)

    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("TOMA 🍅")

    faces = {
        "neutral": pygame.image.load("pic/calm.png"),
        "happy": pygame.image.load("pic/happy.png"),
        "sad": pygame.image.load("pic/sad.png"),
        "fear": pygame.image.load("pic/scared.png"),
        "listening": pygame.image.load("pic/attention.png"),
    }

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if HEAD_RECT.collidepoint(mx, my):
                    print("🍅 被摸头了！")
                    record2.toma_face = "happy"


        screen.fill((255, 255, 255))
        face_key = record2.toma_face
        face = faces.get(face_key, faces["neutral"])
        screen.blit(face, (0, 0))

        pygame.display.flip()
        clock.tick(30)

Thread(target=record2.start_toma, daemon=True).start()

run_pygame()   # 主线程
