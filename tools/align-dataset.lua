#!/usr/bin/env th
--
-- *Face alignment on dataset
--
--   For given (source dataset) path, it traverses all subdirectories and
--   align/crop faces from all images and save them in the destination path.
--
-- Author: Jonghoon Jin
--
require('pl')
require('image')
local FaceAlign = assert(require('libface_align'))

-- Options ---------------------------------------------------------------------
opt = lapp([[
--src      (default 'input')   Directory for input files
--dst      (default 'ouput')   Directory for output files
]])

torch.setdefaulttensortype('torch.FloatTensor')


local use_loose_crop = true


-- set parameters:
local arg = {
   index = 0,    -- index of face
   size = 224,   -- output size (size x size)
}

if use_loose_crop then
   arg.pad = 0.5           -- padding around face [0 1]
   arg.shift = 0.05        -- shift face down in rectangle [0 1]
   arg.alignment_radius = 70
else
   arg.pad = 0.1           -- padding around face [0 1]
   arg.shift = 0           -- shift face down in rectangle [0 1]
   arg.alignment_radius = 50
end


-- check if given path is an image
local function has_image_extension(filename)
   -- grab extension in lowercase letter
   local ext = string.match(string.lower(filename), "%.(%w+)")

   -- compare with list of image extensions
   local img_extensions = {'jpeg', 'jpg', 'png', 'ppm', 'pgm'}
   for i = 1, #img_extensions do
      if ext == img_extensions[i] then
         return true
      end
   end

   -- return false if undefined
   return false
end


-- main routine being called recursively
local function IterateTwoLevelDirectories(dst, src, arg, cnt)
   local cnt = 0

   -- create dst directory
   paths.mkdir(dst)

   -- iter dirs
   for dirc in paths.iterdirs(src) do
      local dirc_src = path.join(src, dirc)
      local dirc_dst = path.join(dst, dirc)
      paths.mkdir(dirc_dst)

      -- iter files
      for file in paths.iterfiles(dirc_src) do
         collectgarbage()

         -- check if image
         if has_image_extension(file) then
            local img_src = path.join(dirc_src, file)
            local img_dst = path.join(dirc_dst, file)

            local img_raw = image.load(img_src)
            img_raw = ((img_raw:size(1) == 1) and img_raw:repeatTensor(3,1,1)) or img_raw

            if FaceAlign.facealign(img_raw) >= 1 then
               local img_align = torch.Tensor()
               FaceAlign.getimage(img_align, arg.index, arg.size, arg.pad, arg.shift)
               image.save(img_dst, image.scale(img_align, arg.size, arg.size))
               cnt = cnt + 1
            else
               print('==> WARNING: fails to find a face in ('..img_src..')')
               local r = arg.alignment_radius
               local cy = math.floor(img_raw:size(2)/2)
               local cx = math.floor(img_raw:size(3)/2)
               image.save(img_dst, image.scale(img_raw[{{1,3},{cy-r,cy+r},{cx-r,cy+r}}], arg.size, arg.size))
               cnt = cnt + 1
            end

         -- skip alignment for non-image file
         else
            print('==> WARNING: skip non-image file ('..path.join(dirc_src, file)..')')
         end
      end
   end
   return cnt
end


-- load model
FaceAlign.loadmodel('shape_predictor_68_face_landmarks.dat')


-- process all images recursively
local timer = torch.Timer()
print('==> Pre-processing images in: ', opt.src)
local nb_img = IterateTwoLevelDirectories(opt.dst, opt.src, arg)
print('==> time elapsed [ms]:', timer:time().real*1000)
print('==> processed images:', nb_img)
