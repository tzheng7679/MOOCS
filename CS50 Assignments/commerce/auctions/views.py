from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from .models import *

def index(request):
    if request.user.is_authenticated:
        alistings = Listing.objects.all()
        
        return render(request, "auctions/index.html", {
            "alistings": alistings})
    else:
        return render(request, "auctions/index.html")


def login_view(request):
    if request.method == "POST":

        # Attempt to sign user in
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)

        # Check if authentication successful
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse("index"))
        else:
            return render(request, "auctions/login.html", {
                "message": "Invalid username and/or password."
            })
    else:
        return render(request, "auctions/login.html")


def logout_view(request):
    logout(request)
    return HttpResponseRedirect(reverse("index"))


def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]

        # Ensure password matches confirmation
        password = request.POST["password"]
        confirmation = request.POST["confirmation"]
        if password != confirmation:
            return render(request, "auctions/register.html", {
                "message": "Passwords must match."
            })

        # Attempt to create new user
        try:
            user = User.objects.create_user(username, email, password)
            user.save()
        except IntegrityError:
            return render(request, "auctions/register.html", {
                "message": "Username already taken."
            })
        login(request, user)
        return HttpResponseRedirect(reverse("index"))
    else:
        return render(request, "auctions/register.html")

def create(request):
    if request.method == "POST":
        name = request.POST["itemname"]
        description = request.POST["itemdescription"]
        price = request.POST["itemprice"]
        image = request.POST["image"]
        catagory = request.POST["catagory"]
        
        if len(image) == 0:
            image = None

        if catagory == "N/A":
            catagory = None
        
        try:
            x = Listing(seller = request.user, price = price, description = description, item = name, image = image, catagory = catagory)
            x.save()
            print(x)
        except Exception as e:
            return render(request, "auctions/create.html", {
                "message": "Invalid"})
        return render(request, "auctions/index.html")
    else:
        return render(request, "auctions/create.html")

def listing(request, id):
    # l is the item being bid on
    l = Listing.objects.filter(pk=id)

    #if object of id dne, then return error page
    if len(l) == 0:
        return render(request, "auctions/error.html")

    #--------------------------------------------------#
    #find the minimum bid that can be made
    l = l.first()
    b = Bid.objects.filter(item = l)

    max = None
    minbid = l.price

    #if there are no bids on the item
    if len(b) == 0:
        b = None

    else:
        max = b.first().price
        for bid in b:
            if bid.price > max:
                max = bid.price

        if max > minbid:
            minbid = float(max) + .01

    if request.method == "POST":
        print(request.POST["bidprice"])
        newbid = Bid(bidder=request.user, item=l, price=request.POST["bidprice"])
        newbid.save()
        return HttpResponseRedirect(reverse("index"))
    else:
        return render(request, "auctions/listing.html", {
            "l" : l,
            "b" : b,
            "max" : max,
            "minbid" : minbid
            })