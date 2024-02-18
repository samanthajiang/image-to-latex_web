from pix2tex.dataset.transforms import test_transform
import pandas.io.clipboard as clipboard
from PIL import ImageGrab
from PIL import Image
import os
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import atexit
from contextlib import suppress
import logging
import yaml
import re

with suppress(ImportError, AttributeError):
    import readline

import numpy as np
import torch
from torch._appdirs import user_data_dir
from munch import Munch
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

from pix2tex.dataset.latex2png import tex2pil
from pix2tex.models import get_model
from pix2tex.utils import *
from pix2tex.model.checkpoints.get_latest_checkpoint import download_checkpoints


def minmax_size(img: Image, max_dimensions: Tuple[int, int] = None, min_dimensions: Tuple[int, int] = None) -> Image:
    """Resize or pad an image to fit into given dimensions

    Args:
        img (Image): Image to scale up/down.
        max_dimensions (Tuple[int, int], optional): Maximum dimensions. Defaults to None.
        min_dimensions (Tuple[int, int], optional): Minimum dimensions. Defaults to None.

    Returns:
        Image: Image with correct dimensionality
    """
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
        padded_size = [max(img_dim, min_dim) for img_dim, min_dim in zip(img.size, min_dimensions)]
        if padded_size != list(img.size):  # assert hypothesis
            padded_im = Image.new('L', padded_size, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img

def minmax_size_my(img: Image, max_dimensions: Tuple[int, int] = None, min_dimensions: Tuple[int, int] = None) -> Image:
    """Resize or pad an image to fit into given dimensions

    Args:
        img (Image): Image to scale up/down.
        max_dimensions (Tuple[int, int], optional): Maximum dimensions. Defaults to None.
        min_dimensions (Tuple[int, int], optional): Minimum dimensions. Defaults to None.

    Returns:
        Image: Image with correct dimensionality
    """
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
        padded_size = [max(img_dim, min_dim) for img_dim, min_dim in zip(img.size, min_dimensions)]
        if padded_size != list(img.size):  # assert hypothesis
            padded_im = Image.new('L', padded_size, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


class LatexOCR:
    '''Get a prediction of an image in the easiest way'''

    image_resizer = None
    last_pic = None

    @in_model_path()
    def __init__(self, arguments=None):
        """Initialize a LatexOCR model

        Args:
            arguments (Union[Namespace, Munch], optional): Special model parameters. Defaults to None.
        """
        if arguments is None:
            arguments = Munch({'config': 'settings/config.yaml', 'checkpoint': 'checkpoints/weights.pth', 'no_cuda': True, 'no_resize': False})
        logging.getLogger().setLevel(logging.FATAL)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        with open(arguments.config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.args = parse_args(Munch(params))
        self.args.update(**vars(arguments))
        self.args.wandb = False
        self.args.device = 'cuda' if torch.cuda.is_available() and not self.args.no_cuda else 'cpu'
        if not os.path.exists(self.args.checkpoint):
            download_checkpoints()
        self.model = get_model(self.args)
        self.model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
        self.model.eval()

        if 'image_resizer.pth' in os.listdir(os.path.dirname(self.args.checkpoint)) and not arguments.no_resize:
            self.image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(self.args.max_dimensions)//32, global_pool='avg', in_chans=1, drop_rate=.05,
                                          preact=True, stem_type='same', conv_layer=StdConv2dSame).to(self.args.device)
            self.image_resizer.load_state_dict(torch.load(os.path.join(os.path.dirname(self.args.checkpoint), 'image_resizer.pth'), map_location=self.args.device))
            self.image_resizer.eval()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.args.tokenizer)

    @in_model_path()
    def __call__(self, img=None, resize=True) -> str:
        """Get a prediction from an image

        Args:
            img (Image, optional): Image to predict. Defaults to None.
            resize (bool, optional): Whether to call the resize model. Defaults to True.

        Returns:
            str: predicted Latex code
        """
        if type(img) is bool:
            img = None
        if img is None:
            if self.last_pic is None:
                return ''
            else:
                print('\nLast image is: ', end='')
                img = self.last_pic.copy()
        else:
            self.last_pic = img.copy()
        img = minmax_size(pad(img), self.args.max_dimensions, self.args.min_dimensions)
        if (self.image_resizer is not None and not self.args.no_resize) and resize:
            # print("===resize")
            with torch.no_grad():
                input_image = img.convert('RGB').copy()
                r, w, h = 1, input_image.size[0], input_image.size[1]
                for _ in range(10):
                    h = int(h * r)  # height to resize
                    img = pad(minmax_size(input_image.resize((w, h), Image.Resampling.BILINEAR if r > 1 else Image.Resampling.LANCZOS), self.args.max_dimensions, self.args.min_dimensions))
                    # print("img_resizer_or",img.size)
                    t = test_transform(image=np.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
                    w = (self.image_resizer(t.to(self.args.device)).argmax(-1).item()+1)*32
                    logging.info(r, img.size, (w, int(input_image.size[1]*r)))
                    if (w == img.size[0]):
                        break
                    r = w/img.size[0]
        else:
            img = np.array(pad(img).convert('RGB'))
            t = test_transform(image=img)['image'][:1].unsqueeze(0)
        im = t.to(self.args.device)

        dec = self.model.generate(im.to(self.args.device), temperature=self.args.get('temperature', .25))
        pred = post_process(token2str(dec, self.tokenizer)[0])
        try:
            clipboard.copy(pred)
        except:
            pass
        return pred

def find_all_left_or_right(latex, left_or_right='left'):
    left_bracket_infos = []
    prefix_len = len(left_or_right) + 1
    # 匹配出latex中所有的 '\left' 后面跟着的第一个非空格字符，定位它们所在的位置
    for m in re.finditer(rf'\\{left_or_right}\s*\S', latex):
        start, end = m.span()
        # 如果最后一个字符为 "\"，则往前继续匹配，直到匹配到一个非字母的字符
        # 如 "\left \big("
        while latex[end - 1] in ('\\', ' '):
            end += 1
            while end < len(latex) and latex[end].isalpha():
                end += 1
        ori_str = latex[start + prefix_len : end].strip()
        # FIXME: ori_str中可能出现多个 '\left'，此时需要分隔开

        left_bracket_infos.append({'str': ori_str, 'start': start, 'end': end})
        left_bracket_infos.sort(key=lambda x: x['start'])
    return left_bracket_infos

def match_left_right(left_str, right_str):
    """匹配左右括号，如匹配 `\left(` 和 `\right)`。"""
    left_str = left_str.strip().replace(' ', '')[len('left') + 1 :]
    right_str = right_str.strip().replace(' ', '')[len('right') + 1 :]
    # 去掉开头的相同部分
    while left_str and right_str and left_str[0] == right_str[0]:
        left_str = left_str[1:]
        right_str = right_str[1:]

    match_pairs = [
        ('', ''),
        ('(', ')'),
        ('\{', '.'),  # 大括号那种
        ('⟮', '⟯'),
        ('[', ']'),
        ('⟨', '⟩'),
        ('{', '}'),
        ('⌈', '⌉'),
        ('┌', '┐'),
        ('⌊', '⌋'),
        ('└', '┘'),
        ('⎰', '⎱'),
        ('lt', 'gt'),
        ('lang', 'rang'),
        (r'langle', r'rangle'),
        (r'lbrace', r'rbrace'),
        ('lBrace', 'rBrace'),
        (r'lbracket', r'rbracket'),
        (r'lceil', r'rceil'),
        ('lcorner', 'rcorner'),
        (r'lfloor', r'rfloor'),
        (r'lgroup', r'rgroup'),
        (r'lmoustache', r'rmoustache'),
        (r'lparen', r'rparen'),
        (r'lvert', r'rvert'),
        (r'lVert', r'rVert'),
    ]
    return (left_str, right_str) in match_pairs

def post_post_process_latex(latex: str) -> str:
    """对识别结果做进一步处理和修正。"""
    # 把latex中的中文括号全部替换成英文括号
    latex = latex.replace('（', '(').replace('）', ')')
    # 把latex中的中文逗号全部替换成英文逗号
    latex = latex.replace('，', ',')

    left_bracket_infos = find_all_left_or_right(latex, left_or_right='left')
    right_bracket_infos = find_all_left_or_right(latex, left_or_right='right')
    # left 和 right 找配对，left找位置比它靠前且最靠近他的right配对
    for left_bracket_info in left_bracket_infos:
        for right_bracket_info in right_bracket_infos:
            if (
                not right_bracket_info.get('matched', False)
                and right_bracket_info['start'] > left_bracket_info['start']
                and match_left_right(
                    right_bracket_info['str'], left_bracket_info['str']
                )
            ):
                left_bracket_info['matched'] = True
                right_bracket_info['matched'] = True
                break

    for left_bracket_info in left_bracket_infos:
        # 把没有匹配的 '\left'替换为等长度的空格
        left_len = len('left') + 1
        if not left_bracket_info.get('matched', False):
            start_idx = left_bracket_info['start']
            end_idx = start_idx + left_len
            latex = (
                latex[: left_bracket_info['start']]
                + ' ' * (end_idx - start_idx)
                + latex[end_idx:]
            )
    for right_bracket_info in right_bracket_infos:
        # 把没有匹配的 '\right'替换为等长度的空格
        right_len = len('right') + 1
        if not right_bracket_info.get('matched', False):
            start_idx = right_bracket_info['start']
            end_idx = start_idx + right_len
            latex = (
                latex[: right_bracket_info['start']]
                + ' ' * (end_idx - start_idx)
                + latex[end_idx:]
            )

    # 把 latex 中的连续空格替换为一个空格
    latex = re.sub(r'\s+', ' ', latex)
    return latex


class LatexOCR_my:
    '''Get a prediction of an image in the easiest way'''

    image_resizer = None
    last_pic = None

    @in_model_path()
    def __init__(self, arguments=None):
        """Initialize a LatexOCR model

        Args:
            arguments (Union[Namespace, Munch], optional): Special model parameters. Defaults to None.
        """
        if arguments is None:
            arguments = Munch({'config': 'settings/config.yaml', 'checkpoint': 'checkpoints/weights.pth', 'no_cuda': True, 'no_resize': False})
        logging.getLogger().setLevel(logging.FATAL)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        with open(arguments.config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.args = parse_args(Munch(params))
        self.args.update(**vars(arguments))
        self.args.wandb = False
        self.args.device = 'cuda' if torch.cuda.is_available() and not self.args.no_cuda else 'cpu'
        if not os.path.exists(self.args.checkpoint):
            download_checkpoints()
        self.model = get_model(self.args)
        # todo
        self.args.checkpoint = 'checkpoints/weights_176.pth'
        self.model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
        # print("===self.model.load_state_dict",self.args.checkpoint)
        self.model.eval()

        if 'image_resizer.pth' in os.listdir(os.path.dirname(self.args.checkpoint)) and not arguments.no_resize:
            # print("===self.image_resizer")
            self.image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(self.args.max_dimensions)//32, global_pool='avg', in_chans=1, drop_rate=.05,
                                          preact=True, stem_type='same', conv_layer=StdConv2dSame).to(self.args.device)
            self.image_resizer.load_state_dict(torch.load(os.path.join(os.path.dirname(self.args.checkpoint), 'image_resizer.pth'), map_location=self.args.device))
            self.image_resizer.eval()
        # todo
        self.args.tokenizer = 'dataset/tokenizer_my.json'
        # print("self.args.tokenizer",self.args.tokenizer)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.args.tokenizer)

    @in_model_path()
    def __call__(self, img=None, resize=True) -> str:
        """Get a prediction from an image

        Args:
            img (Image, optional): Image to predict. Defaults to None.
            resize (bool, optional): Whether to call the resize model. Defaults to True.

        Returns:
            str: predicted Latex code
        """
        if type(img) is bool:
            img = None
        if img is None:
            if self.last_pic is None:
                return ''
            else:
                print('\nLast image is: ', end='')
                img = self.last_pic.copy()
        else:
            self.last_pic = img.copy()
        # todo
        img = pad_my(img)

        # # todo
        # img = minmax_size(img, self.args.max_dimensions, self.args.min_dimensions)
        # todo
        # resize = False
        if (self.image_resizer is not None and not self.args.no_resize) and resize:
            # print("===do_resize")
            with torch.no_grad():
                input_image = img.convert('RGB').copy()
                r, w, h = 1, input_image.size[0], input_image.size[1]
                # print("input_image.size[0], input_image.size[1]",input_image.size[0], input_image.size[1])
                for _ in range(10):
                    h = int(h * r)  # height to resize
                    img = pad(minmax_size(input_image.resize((w, h), Image.Resampling.BILINEAR if r > 1 else Image.Resampling.LANCZOS), self.args.max_dimensions, self.args.min_dimensions))
                    # print("===img_resizer_my",img.size)
                    # img.show()
                    t = test_transform(image=np.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
                    w = (self.image_resizer(t.to(self.args.device)).argmax(-1).item()+1)*32
                    logging.info(r, img.size, (w, int(input_image.size[1]*r)))
                    # print("w",w)
                    # print("img.size[0]",img.size[0])
                    if (w == img.size[0]):
                        break
                    r = w/img.size[0]
        else:
            # img = np.array(pad(img).convert('RGB'))
            img = np.array(img.convert('RGB'))
            t = test_transform(image=img)['image'][:1].unsqueeze(0)
        im = t.to(self.args.device)

        dec = self.model.generate(im.to(self.args.device), temperature=self.args.get('temperature', .25))
        pred = post_process(token2str(dec, self.tokenizer)[0])
        # print("===pred\n",pred)
        # # todo
        pred = post_post_process_latex(pred)
        # print("===post_post_process_latex\n", pred)
        try:
            clipboard.copy(pred)
        except:
            pass
        return pred

class LatexOCR_my_non_resize:
    '''Get a prediction of an image in the easiest way'''

    image_resizer = None
    last_pic = None

    @in_model_path()
    def __init__(self, arguments=None):
        """Initialize a LatexOCR model

        Args:
            arguments (Union[Namespace, Munch], optional): Special model parameters. Defaults to None.
        """
        if arguments is None:
            arguments = Munch({'config': 'settings/config.yaml', 'checkpoint': 'checkpoints/weights.pth', 'no_cuda': True, 'no_resize': False})
        logging.getLogger().setLevel(logging.FATAL)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        with open(arguments.config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.args = parse_args(Munch(params))
        self.args.update(**vars(arguments))
        self.args.wandb = False
        self.args.device = 'cuda' if torch.cuda.is_available() and not self.args.no_cuda else 'cpu'
        if not os.path.exists(self.args.checkpoint):
            download_checkpoints()
        self.model = get_model(self.args)
        # todo
        self.args.checkpoint = 'checkpoints/weights_176.pth'
        # print("self.args.checkpoint",self.args.checkpoint)
        self.model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
        # print("===self.model.load_state_dict",self.args.checkpoint)
        self.model.eval()

        # if 'image_resizer.pth' in os.listdir(os.path.dirname(self.args.checkpoint)) and not arguments.no_resize:
        #     print("===self.image_resizer")
        #     self.image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(self.args.max_dimensions)//32, global_pool='avg', in_chans=1, drop_rate=.05,
        #                                   preact=True, stem_type='same', conv_layer=StdConv2dSame).to(self.args.device)
        #     self.image_resizer.load_state_dict(torch.load(os.path.join(os.path.dirname(self.args.checkpoint), 'image_resizer.pth'), map_location=self.args.device))
        #     self.image_resizer.eval()
        # todo
        self.args.tokenizer = 'dataset/tokenizer_my.json'
        # print("self.args.tokenizer",self.args.tokenizer)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.args.tokenizer)

    @in_model_path()
    def __call__(self, img=None, resize=True) -> str:
        """Get a prediction from an image

        Args:
            img (Image, optional): Image to predict. Defaults to None.
            resize (bool, optional): Whether to call the resize model. Defaults to True.

        Returns:
            str: predicted Latex code
        """
        if type(img) is bool:
            img = None
        if img is None:
            if self.last_pic is None:
                return ''
            else:
                print('\nLast image is: ', end='')
                img = self.last_pic.copy()
        else:
            self.last_pic = img.copy()
        # todo
        img = pad_my(img)
        # print("img",img.size)
        # # todo
        # img = minmax_size(img, self.args.max_dimensions, self.args.min_dimensions)
        # todo
        resize = False
        if (self.image_resizer is not None and not self.args.no_resize) and resize:
            # print("===do_resize")
            with torch.no_grad():
                input_image = img.convert('RGB').copy()
                r, w, h = 1, input_image.size[0], input_image.size[1]
                # print("input_image.size[0], input_image.size[1]",input_image.size[0], input_image.size[1])
                for _ in range(10):
                    h = int(h * r)  # height to resize
                    img = pad_my(minmax_size(input_image.resize((w, h), Image.Resampling.BILINEAR if r > 1 else Image.Resampling.LANCZOS), self.args.max_dimensions, self.args.min_dimensions))
                    t = test_transform(image=np.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
                    w = (self.image_resizer(t.to(self.args.device)).argmax(-1).item()+1)*32
                    logging.info(r, img.size, (w, int(input_image.size[1]*r)))
                    if (w == img.size[0]):
                        break
                    r = w/img.size[0]
        else:
            # img = np.array(pad(img).convert('RGB'))
            img = np.array(img.convert('RGB'))
            t = test_transform(image=img)['image'][:1].unsqueeze(0)
        im = t.to(self.args.device)

        dec = self.model.generate(im.to(self.args.device), temperature=self.args.get('temperature', .25))
        pred = post_process(token2str(dec, self.tokenizer)[0])
        # print("===pred\n",pred)
        # # todo
        pred = post_post_process_latex(pred)
        # print("===post_post_process_latex\n", pred)
        try:
            clipboard.copy(pred)
        except:
            pass
        return pred


def output_prediction(pred, args):
    TERM = os.getenv('TERM', 'xterm')
    if not sys.stdout.isatty():
        TERM = 'dumb'
    try:
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import get_formatter_by_name

        if TERM.split('-')[-1] == '256color':
            formatter_name = 'terminal256'
        elif TERM != 'dumb':
            formatter_name = 'terminal'
        else:
            formatter_name = None
        if formatter_name:
            formatter = get_formatter_by_name(formatter_name)
            lexer = get_lexer_by_name('tex')
            # print(highlight(pred, lexer, formatter), end='')
    except ImportError:
        TERM = 'dumb'
    if TERM == 'dumb':
        print(pred)
    if args.show or args.katex:
        try:
            if args.katex:
                raise ValueError
            tex2pil([f'$${pred}$$'])[0].show()
        except Exception as e:
            # render using katex
            import webbrowser
            from urllib.parse import quote
            url = 'https://katex.org/?data=' + \
                quote('{"displayMode":true,"leqno":false,"fleqn":false,"throwOnError":true,"errorColor":"#cc0000",\
"strict":"warn","output":"htmlAndMathml","trust":false,"code":"%s"}' % pred.replace('\\', '\\\\'))
            webbrowser.open(url)


def predict(model, file, arguments):
    img = None
    if file:
        try:
            img = Image.open(os.path.expanduser(file))
        except Exception as e:
            print(e, end='')
    else:
        try:
            img = ImageGrab.grabclipboard()
        except NotImplementedError as e:
            print(e, end='')
    pred = model(img)
    output_prediction(pred, arguments)

def check_file_path(paths:List[Path], wdir:Optional[Path]=None)->List[str]:
    files = []
    for path in paths:
        if type(path)==str:
            if path=='':
                continue
            path=Path(path)
        pathsi = ([path] if wdir is None else [path, wdir/path])
        for p in pathsi:
            if p.exists():
                files.append(str(p.resolve()))
            elif '*' in path.name:
                files.extend([str(pi.resolve()) for pi in p.parent.glob(p.name)])
    return list(set(files))

def main(arguments):
    path = user_data_dir('pix2tex')
    os.makedirs(path, exist_ok=True)
    history_file = os.path.join(path, 'history.txt')
    with suppress(NameError):
        # user can `ln -s /dev/null ~/.local/share/pix2tex/history.txt` to
        # disable history record
        with suppress(OSError):
            readline.read_history_file(history_file)
        atexit.register(readline.write_history_file, history_file)
    files = check_file_path(arguments.file)
    wdir = Path(os.getcwd())
    with in_model_path():
        model = LatexOCR(arguments)
        if files:
            for file in check_file_path(arguments.file, wdir):
                print(file + ': ', end='')
                predict(model, file, arguments)
                model.last_pic = None
                with suppress(NameError):
                    readline.add_history(file)
            exit()
        pat = re.compile(r't=([\.\d]+)')
        while True:
            try:
                instructions = input('Predict LaTeX code for image ("h" for help). ')
            except KeyboardInterrupt:
                # TODO: make the last line gray
                print("")
                continue
            except EOFError:
                break
            file = instructions.strip()
            ins = file.lower()
            t = pat.match(ins)
            if ins == 'x':
                break
            elif ins in ['?', 'h', 'help']:
                print('''pix2tex help:

    Usage:
        On Windows and macOS you can copy the image into memory and just press ENTER to get a prediction.
        Alternatively you can paste the image file path here and submit.

        You might get a different prediction every time you submit the same image. If the result you got was close you
        can just predict the same image by pressing ENTER again. If that still does not work you can change the temperature
        or you have to take another picture with another resolution (e.g. zoom out and take a screenshot with lower resolution). 

        Press "x" to close the program.
        You can interrupt the model if it takes too long by pressing Ctrl+C.

    Visualization:
        You can either render the code into a png using XeLaTeX (see README) to get an image file back.
        This is slow and requires a working installation of XeLaTeX. To activate type 'show' or set the flag --show
        Alternatively you can render the expression in the browser using katex.org. Type 'katex' or set --katex

    Settings:
        to toggle one of these settings: 'show', 'katex', 'no_resize' just type it into the console
        Change the temperature (default=0.333) type: "t=0.XX" to set a new temperature.
                    ''')
                continue
            elif ins in ['show', 'katex', 'no_resize']:
                setattr(arguments, ins, not getattr(arguments, ins, False))
                print('set %s to %s' % (ins, getattr(arguments, ins)))
                continue
            elif t is not None:
                t = t.groups()[0]
                model.args.temperature = float(t)+1e-8
                print('new temperature: T=%.3f' % model.args.temperature)
                continue
            files = check_file_path(file.split(' '), wdir)
            with suppress(KeyboardInterrupt):
                if files:
                    for file in files:
                        if len(files)>1:
                            print(file + ': ', end='')
                        predict(model, file, arguments)
                else:
                    predict(model, file, arguments)
            file = None
