from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    pass

class Listing(models.Model):
    seller = models.ForeignKey(User, on_delete=models.CASCADE, related_name="listings")
    item = models.CharField(max_length=100)
    description = models.CharField(max_length=500, default="No description")
    #image is url for image
    image = models.CharField(max_length = 200, blank = True)
    price = models.DecimalField(max_digits=6, decimal_places=2)
    catagory = models.CharField(max_length = 50, blank = True)
    def __str__(self):
        return f"{self.item} (id = {self.id}) sold by {self.seller} for ${self.price}"

class Bid(models.Model):
    bidder = models.ForeignKey(User, on_delete=models.CASCADE, related_name="bids")
    item = models.ForeignKey(Listing, on_delete=models.CASCADE, related_name="bids")
    price = models.DecimalField(max_digits=6, decimal_places=2)

    def __str__(self):
        return f"bid on {self.item} by {self.bidder} for ${self.price}"

class Comment(models.Model):
    commenter = models.ForeignKey(User, on_delete=models.CASCADE, related_name="comments")
    text = models.CharField(max_length=250)
    item = models.ForeignKey(Listing, on_delete=models.CASCADE, related_name="comments")

    def __str__(self):
        return f"{self.commenter} comments \"{self.text}\" on {self.item}"