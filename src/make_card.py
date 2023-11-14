from PIL import Image

card_bgs = {
    card_name: Image.open(f"./assets/{card_name.lower()}-trans.png", "r")
    for card_name in ["death", "magician", "priestess", "sun", "world"]
}


def make_card(image: Image.Image, card_name: str) -> Image.Image:
    """
    Make a Tarot card from an image and text.
    """
    card = card_bgs[card_name.lower()]
    card_w, card_h = 620, 868
    card = card.resize((card_w, card_h))

    # Paste image onto black bg
    im = Image.new(mode="RGB", size=(card_w, card_h))

    side_margin = int(card_w / 2 - image.size[0] / 2)
    im.paste(image, (side_margin, side_margin))

    # finally paste transparent card onto image
    im.paste(card, (0, 0), card.split()[-1])

    return im
