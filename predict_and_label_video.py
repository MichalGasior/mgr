from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import io
import ffmpeg

pipe = pipeline("image-classification", 
                model='MichalGas/vit-base-patch16-224-in21k-finetuned-mgasior-07-02-2024')



in_filename = 'input_vid/SBA_skin_to_skin.mp4'
p = ffmpeg.probe(in_filename, select_streams='v');
width = p['streams'][0]['width']
height = p['streams'][0]['height']

process1 = (
    ffmpeg
    .input(in_filename).filter('setpts', '0.1*PTS')
    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    .run_async(pipe_stdout=True)
)
process2 = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
    .output('processed.mp4', pix_fmt='yuv420p')
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

font = ImageFont.load_default(size=30)

while True:
    data = process1.stdout.read(width*height*3)
    if not data:
        break   
    image = Image.frombytes('RGB',(width,height), io.BytesIO(data).read())
    res = pipe(image)
    best = res[0]
    multi =  [l for l in res if l['score'] > 0.30] 
    pick =  [best] if len(multi) < 2 else multi
    labels = [l['label'] for l in pick]
    labels = sorted(labels)
    # print(res)
    # print(labels)

    draw = ImageDraw.Draw(image)
    draw.text((0, 0),' '.join(labels), (255,255,255), font=font)
    process2.stdin.write(
        image.tobytes()
    )

process2.stdin.close()
process1.wait()
process2.wait()


