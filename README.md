# GiGiLive
Software for live stacking of astronomy deep sky pictures. It works with INDI and EKOS 
-------------------------------------------------------------------------------------------------------

How to use:
--------------
1. Lauch KStars
2. Tools -> Ekos -> start Ekos
3. Launch GiGiLive
4. web browser: http://127.0.0.1:5000/adminsecretpage
    http://127.0.0.1:5000/ for the audience (you'll need to set up a network from the controlling PC)

Features:
-----------
Aligns and stacks in real time 
accessible via browser
separate webpages for admin (controls available) and audience (just see)

Libraries:
-----------
* astropy.io
* PyIndi
* flask
