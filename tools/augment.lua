-- Author: Aysegul Dundar and Sangpil Kim
-- Date: December, 2014 and May, 2016
require 'image'
require 'trepl'
require 'paths'

function hflip_img(dst, img)
  local img_flip = image.hflip(img)
  -- assumes .png ending for now
  local img_path = string.sub(dst, 1, -5) .. "hflip.png"
  image.save(img_path, img_flip)

end

function vflip_img(dst, img)
  local img_flip = image.vflip(img)
  -- assumes .png ending for now
  local img_path = string.sub(dst,1, -5) .. "vflip.png"
  image.save(img_path, img_flip)

end

function rotate_img(flip, dst, degree, img)

  local img_rotate = image.rotate(img, degree,'bilinear')
  -- assumes .png ending for now
  local img_path = string.sub(dst,1, -5)  ..  "rotate" .. tostring(degree*10) .. ".png"
  image.save(img_path, img_rotate)
  if flip == true then
     hflip_img(img_path,img_rotate)
  end

end


function crop5_img(flip, dst ,img , max_trans, transImgInt)

  local w = img:size(2) - max_trans
  local h = img:size(3) - max_trans
  -- assumes .png ending for now

  local number_img = max_trans/transImgInt

  for i=1, number_img do
    for j=1, number_img do
       local sample = img[{{}, {1+ (j-1) * transImgInt, (j-1) * transImgInt + w},
                               {1+ (i-1) * transImgInt, (i-1) * transImgInt + h}}]

       local img_path = string.sub(dst,1,-5) .. "crop" .. (j+(i-1)*number_img) .. ".png"
       image.save(img_path, sample)


       if flip == true then
         hflip_img(img_path,sample)
       end

     end
   end

end
function blend(img1, img2, alpha)
   return img1:mul(alpha):add(1 - alpha, img2)
end
function grayscale(img,dst)
   img[2]:copy(img[1])
   img[3]:copy(img[1])
   if dst then
      local img_path = string.sub(dst,1,-5) .. "gray" .. sat .. ".png"
      image.save(img_path, img)
   end
end
function Saturation(sat, img, dst)
	img[1]:add(sat)
	img[2]:add(sat)
	img[3]:add(sat)
	local img_path = string.sub(dst,1,-5) .. "sat" .. sat .. ".png"
	image.save(img_path, img)
end
function Contrast(con,img,dst)
   gs = img:clone()
   grayscale(gs)
   gs:fill(gs[1]:mean())
   con = 0.7
	local alpha = 1.0 + torch.uniform(-con, con)
	sample = blend(img, gs, alpha)
	local img_path = string.sub(dst,1,-5) .. "cont" .. con .. ".png"
	image.save(img_path, sample)
end
function ColorJitter(r, g, b, img, dst)
	img[1]:add(r)
	img[2]:add(g)
	img[3]:add(b)
	local img_path = string.sub(dst,1,-5) .. "color".. ".png"
	image.save(img_path, img)
end
