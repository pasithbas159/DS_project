import arcade

#CONSTANT
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
SCREEN_TITLE = "b1a5s9 Game"

PLAYER_SCALING = 1
TILE_SCALING = 0.5
COIN_SCALING = 0.5
LASER_SCALING = 0.2

PLAYER_MOVEMENT_SPEED = 5
PLAYER_JUMP_SPEED = 20
BULLET_SPEED = 6
GRAVITY = 1

LEFT_VIEWPORT_MARGIN = 150
RIGHT_VIEWPORT_MARGIN = 150
BOTTOM_VIEWPORT_MARGIN = 50
TOP_VIEWPORT_MARGIN = 100

PLAYER_START_X = 90
PLAYER_START_Y = 120

HEALTH = 100
SCORE = 0

health_point = []
score_point = []


class MenuView(arcade.View):
    """ Class that manages the 'menu' view. """

    def on_show(self):
        """ Called when switching to this view"""
        arcade.set_background_color(arcade.color.WHITE)

    def on_draw(self):
        """ Draw the menu """
        arcade.start_render()
        arcade.draw_text("Welcome to b1a5s9 Game!! - Enter to start", SCREEN_WIDTH / 2,
                         SCREEN_HEIGHT / 2, arcade.color.BLACK, font_size=30, anchor_x="center")

    def on_key_press(self, key, _modifiers):
        """ If user hits escape, go back to the main menu view """
        if key == arcade.key.ENTER:
            game_view = MyGame()
            game_view.setup()
            self.window.show_view(game_view)


class MyGame(arcade.View):
    def __init__(self):
        super().__init__()
        arcade.set_background_color(arcade.csscolor.GREY)

        self.player_sprite = None
        self.player_list = None
        self.wall_list = None
        self.spike_list = None
        self.coin_list = None
        self.enemy_list = None
        self.bullet_list = None
        self.flag_list = None
        # self.ladder_list = None

        self.physics_engine = None

        self.view_bottom = 0
        self.view_left = 0

        self.score = 0
        self.health = 100

        self.game_over = False
        self.game_win = False
        # self.set_mouse_visible(False)

    def setup(self):

        self.player_list = arcade.SpriteList()
        self.wall_list = arcade.SpriteList(use_spatial_hash=True)
        self.spike_list = arcade.SpriteList(use_spatial_hash=True)
        self.coin_list = arcade.SpriteList(use_spatial_hash=True)
        self.enemy_list = arcade.SpriteList(use_spatial_hash=True)
        self.bullet_list = arcade.SpriteList(use_spatial_hash=True)
        self.flag_list = arcade.SpriteList(use_spatial_hash=True)
        # self.ladder_list = arcade.SpriteList(use_spatial_hash=True)

        self.view_bottom = 0
        self.view_left = 0

        self.player_sprite = arcade.Sprite("images/player_1/player_stand_w_gun.png", PLAYER_SCALING)
        self.player_sprite.center_x = PLAYER_START_X
        self.player_sprite.center_y = PLAYER_START_Y
        self.player_list.append(self.player_sprite)

        for x in range(0, 6000, 64):
            wall = arcade.Sprite("images/tiles/sandCenter.png", TILE_SCALING)
            wall.center_x = x
            wall.center_y = 32
            self.wall_list.append(wall)

        for crate in range(333, 4800, 333):
            wall = arcade.Sprite("images/tiles/cactus.png", TILE_SCALING + 0.15)
            wall.center_x = crate
            wall.center_y = 105
            self.spike_list.append(wall)

        for x in range(333, 4800, 333):
            coin = arcade.Sprite("images/items/coinGold.png", COIN_SCALING)
            coin.center_x = x
            coin.center_y = 155
            self.coin_list.append(coin)

        for x in range(444, 4800, 999):
            enemy = arcade.Sprite("images/enemies/wormGreen.png", TILE_SCALING + 0.25)
            enemy.bottom = 65
            enemy.left = x
            # Set enemy initial speed
            enemy.change_x = 2
            self.enemy_list.append(enemy)

        coordinate = [5000, 90]
        flag = arcade.Sprite("images/items/flagYellow1.png", TILE_SCALING)
        flag.position = coordinate
        self.flag_list.append(flag)

        # for y in range(200, 601, 64):
        #     ladder = arcade.Sprite("images/tiles/ladderMid.png", TILE_SCALING)
        #     ladder.center_x = 333
        #     ladder.center_y = y
        #     self.ladder_list.append(ladder)

        self.physics_engine = arcade.PhysicsEnginePlatformer(self.player_sprite, self.wall_list, GRAVITY)
                                                             # ladders=self.ladder_list

    def on_key_press(self, key, modifiers):
        if key == arcade.key.UP or key == arcade.key.W:
            if self.physics_engine.can_jump():
                self.player_sprite.change_y = PLAYER_JUMP_SPEED
            elif self.physics_engine.is_on_ladder():
                self.player_sprite.change_y = PLAYER_MOVEMENT_SPEED
        elif key == arcade.key.DOWN or key == arcade.key.S:
            if self.physics_engine.is_on_ladder():
                self.player_sprite.change_y = -PLAYER_MOVEMENT_SPEED
        elif key == arcade.key.LEFT or key == arcade.key.A:
            if self.player_sprite.left < LEFT_VIEWPORT_MARGIN:
                self.player_sprite.change_x = 5
            else:
                self.player_sprite.change_x = -PLAYER_MOVEMENT_SPEED
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.player_sprite.change_x = PLAYER_MOVEMENT_SPEED
        elif key == arcade.key.SPACE:
            bullet = arcade.Sprite("images/items/keyRed.png", LASER_SCALING)
            bullet.change_x = BULLET_SPEED
            bullet.center_x = self.player_sprite.center_x
            bullet.bottom = self.player_sprite.bottom + 30
            self.bullet_list.append(bullet)
            self.bullet_list.change_x = BULLET_SPEED

        if self.game_win:
            if key == arcade.key.ENTER:
                self.player_sprite.center_x = 300
                menu_view = MenuView()
                self.window.show_view(menu_view)

        if self.game_over:
            if key == arcade.key.ENTER:
                self.player_sprite.center_x = 300
                menu_view = MenuView()
                self.window.show_view(menu_view)


    def on_key_release(self, key, modifiers):
        if key == arcade.key.UP or key == arcade.key.W:
            self.player_sprite.change_y = 0
        elif key == arcade.key.DOWN or key == arcade.key.S:
            self.player_sprite.change_y = 0
        elif key == arcade.key.LEFT or key == arcade.key.A:
            self.player_sprite.change_x = 0
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.player_sprite.change_x = 0
        elif key == arcade.key.E:
            self.bullet_list.change_x = 0

    def on_update(self, delta_time: float):

        self.bullet_list.update()

        # Loop through each bullet
        for bullet in self.bullet_list:

            # Check this bullet to see if it hit a coin
            hit_list = arcade.check_for_collision_with_list(bullet, self.spike_list)

            # If it did, get rid of the bullet
            if len(hit_list) > 0:
                bullet.remove_from_sprite_lists()

            # If the bullet flies off-screen, remove it.
            if bullet.bottom > SCREEN_HEIGHT:
                bullet.remove_from_sprite_lists()

            enemy_hit_list = arcade.check_for_collision_with_list(bullet, self.enemy_list)

            if len(enemy_hit_list) > 0:
                bullet.remove_from_sprite_lists()

            if bullet.bottom > SCREEN_HEIGHT:
                bullet.remove_from_sprite_lists()

            for enemy in enemy_hit_list:
                enemy.remove_from_sprite_lists()
                self.score += 1
                return self.score

        if not self.game_over:
            # Move the enemies
            self.enemy_list.update()

            # Check each enemy
            for enemy in self.enemy_list:
                # If the enemy hit a wall, reverse
                if len(arcade.check_for_collision_with_list(enemy, self.spike_list)) > 0:
                    enemy.change_x *= -1
                # If the enemy hit the left boundary, reverse
                elif enemy.boundary_left is not None and enemy.left < enemy.boundary_left:
                    enemy.change_x *= -1
                # If the enemy hit the right boundary, reverse
                elif enemy.boundary_right is not None and enemy.right > enemy.boundary_right:
                    enemy.change_x *= -1

        self.physics_engine.update()

        coin_hit_list = arcade.check_for_collision_with_list(self.player_sprite,
                                                             self.coin_list)

        for coin in coin_hit_list:
            coin.remove_from_sprite_lists()
            self.score += 1
            return self.score

        spike_hit_list = arcade.check_for_collision_with_list(self.player_sprite,
                                                              self.spike_list)

        for obj in spike_hit_list:
            if self.health == 0:
                self.player_sprite.center_x = PLAYER_START_X
                self.player_sprite.center_y = PLAYER_START_Y
                self.game_over = True
                self.view_left = 0
                self.view_bottom = 0
                changed = True
            else:
                self.health -= 1

        enemies_hit_list = arcade.check_for_collision_with_list(self.player_sprite,
                                                              self.enemy_list)

        for obj in enemies_hit_list:
            if self.health == 0:
                self.player_sprite.center_x = PLAYER_START_X
                self.player_sprite.center_y = PLAYER_START_Y
                self.game_over = True
                self.view_left = 0
                self.view_bottom = 0
                changed = True
            else:
                # self.health -= 1
                self.health = self.health - 1
                return self.health

        flag_hit_list = arcade.check_for_collision_with_list(self.player_sprite,
                                                             self.flag_list)

        for obj in flag_hit_list:
            self.player_sprite.center_x = PLAYER_START_X
            self.player_sprite.center_y = PLAYER_START_Y
            self.view_left = 0
            self.view_bottom = 0
            self.game_win = True
            changed = True
            # game_win = GameWinView()
            # game_win.on_draw()
            # self.window.show_view(game_win)

        # Track if we need to change the viewport

        changed = False

        # Scroll left
        left_boundary = self.view_left + LEFT_VIEWPORT_MARGIN
        if self.player_sprite.left < left_boundary:
            self.view_left -= left_boundary - self.player_sprite.left
            changed = True

        # Scroll right
        right_boundary = self.view_left + SCREEN_WIDTH - RIGHT_VIEWPORT_MARGIN - 300
        if self.player_sprite.right > right_boundary:
            self.view_left += self.player_sprite.right - right_boundary
            changed = True

        # Scroll up
        top_boundary = self.view_bottom + SCREEN_HEIGHT - TOP_VIEWPORT_MARGIN
        if self.player_sprite.top > top_boundary:
            self.view_bottom += self.player_sprite.top - top_boundary
            changed = True

        # Scroll down
        bottom_boundary = self.view_bottom + BOTTOM_VIEWPORT_MARGIN
        if self.player_sprite.bottom < bottom_boundary:
            self.view_bottom -= bottom_boundary - self.player_sprite.bottom
            changed = True

        if changed:
            # Only scroll to integers. Otherwise we end up with pixels that
            # don't line up on the screen
            self.view_bottom = int(self.view_bottom)
            self.view_left = int(self.view_left)

            # Do the scrolling
            arcade.set_viewport(self.view_left,
                                SCREEN_WIDTH + self.view_left,
                                self.view_bottom,
                                SCREEN_HEIGHT + self.view_bottom)

        if self.game_over or self.game_win:
            if self.player_sprite.center_x > 250:
                self.player_sprite.center_x = 200
            elif self.player_sprite.center_x < 10:
                self.player_sprite.center_x = 20


    def on_draw(self):

        arcade.start_render()

        self.player_list.draw()
        self.wall_list.draw()
        self.spike_list.draw()
        self.coin_list.draw()
        self.enemy_list.draw()
        self.bullet_list.draw()
        self.flag_list.draw()
        # self.ladder_list.draw()

        score_text = f"Score: {self.score}"
        arcade.draw_text(score_text, 10 + self.view_left, 10 + self.view_bottom,
                         arcade.csscolor.WHITE, 18)

        health_text = f"Health Left: {self.health}"
        arcade.draw_text(health_text, 100 + self.view_left, 10 + self.view_bottom,
                         arcade.csscolor.WHITE, 18)

        if self.game_over:
            arcade.draw_text("GAME OVER!!!" + "\n" + f"score: {self.score}" + "\n" + "ENTER to MENU",
                             250, 300, arcade.color.WHITE, 55)
            # self.set_mouse_visible(True)

        if self.game_win:
            arcade.draw_text("YOU WIN!!!" + "\n" + f"remain health: {self.health}"
                             + "\n" + f"score: {self.score}" + "\n" + "ENTER to RESTART"
                             , 250, 300, arcade.color.WHITE, 55)
            health_point.append(self.health)
            score_point.append(self.score)

    @property
    def getHealth(self):
        return self.health

    @property
    def getScore(self):
        return self.score


"""end game screen (on progress)"""
# class GameWinView(arcade.View):
#     """ Class to manage the game over view """
#     def on_show(self):
#         """ Called when switching to this view"""
#         arcade.set_background_color(arcade.color.BLACK)
#
#     def on_draw(self):
#         """ Draw the game over view """
#         arcade.start_render()
#         # arcade.draw_text("You win!!! - press ESCAPE to advance", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2,
#         #                  arcade.color.WHITE, 30, anchor_x="center")
#         arcade.draw_text("You win!!! - press ESCAPE to advance" + "\n" +
#                          f"Remain health: {MyGame().getHealth}" + "\n" +
#                          f"Remain Score: {MyGame().getScore}", SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2,
#                          arcade.color.WHITE, 30, anchor_x="center")
#
#     def on_key_press(self, key, _modifiers):
#         """ If user hits escape, go back to the main menu view """
#         if key == arcade.key.ESCAPE:
#             menu_view = MenuView()
#             self.window.show_view(menu_view)


def main():
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    menu_view = MenuView()
    window.show_view(menu_view)
    arcade.run()

if __name__ == '__main__':
    main()