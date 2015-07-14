image_formats = ('JPEG', 'PNG', 'PPM')
default_image_format = 'PPM'

default_video_format = 'h264'
default_bit_rate = 2000		 # kbits/sec
default_quality = 'good'	 # Variable bit rate quality

qualities = ('low', 'fair', 'medium', 'good', 'high', 'higher', 'highest')

# Video formats

formats = {}
h264_quality = {'option_name':'-crf',
		'low':28, 'fair':25, 'medium':23, 'good':20,
		'high':18, 'higher':17, 'highest':15}
formats['h264'] = {'label': 'H.264',
			 'suffix': 'mp4',
			 'ffmpeg_name': 'mp4',
			 'ffmpeg_codec': 'libx264',
			 'ffmpeg_quality': h264_quality,
			 'limited_framerates': False,
			 'size_restriction': (2,2),
			 }
vp8_quality = {'option_name':'-vb',
	       'low':'200k', 'fair':'500k', 'medium':'2000k',
	       'good':'4000k', 'high':'8000k', 'higher':'12000k',
	       'highest': '25000k'}
formats['vp8'] = {'label': 'VP8/WebM',
			'suffix': 'webm',
			'ffmpeg_name': 'webm',
			'ffmpeg_codec': 'libvpx',
			'ffmpeg_quality': vp8_quality,
			'limited_framerates': False,
			'size_restriction': None,
			}
theora_quality = {'option_name':'-qscale',
		  'low':1, 'fair':3, 'medium':5, 'good':6,
		  'high':8, 'higher':9, 'highest':10}
formats['theora'] = {'label': 'Theora',
			   'suffix': 'ogv',
			   'ffmpeg_name': 'ogg',
			   'ffmpeg_codec': 'libtheora',
			   'ffmpeg_quality': theora_quality,
			   'limited_framerates': False,
			   'size_restriction': None,
			   }
mpeg_quality = {'option_name':'-qscale',
		'low':25, 'fair':15, 'medium':10, 'good':8,
		'high':5, 'higher':3, 'highest':1}
formats['mov'] = {'label': 'Quicktime',
			'suffix': 'mov',
			'ffmpeg_name': 'mov',
			'ffmpeg_codec': 'mpeg4',
			'ffmpeg_quality': mpeg_quality,
			'limited_framerates': False,
			'size_restriction': None,
			}
formats['avi'] = {'label': 'AVI MSMPEG-4v2',
			'suffix': 'avi',
			'ffmpeg_name': 'avi',
			'ffmpeg_codec': 'msmpeg4v2',
			'ffmpeg_quality': mpeg_quality,
			'limited_framerates': False,
			'size_restriction': None,
			}
formats['mp4'] = {'label': 'MPEG-4',
			'suffix': 'mpeg4',
			'ffmpeg_name': 'mp4',
			'ffmpeg_codec': None,
			'ffmpeg_quality': mpeg_quality,
			'limited_framerates': False,
			 'size_restriction': (2,2),
			}
formats['mp2'] = {'label': 'MPEG-2',
			'suffix': 'mpg',
			'ffmpeg_name': 'mpeg2video',
			'ffmpeg_codec': None,
			'ffmpeg_quality': mpeg_quality,
			'limited_framerates': True,
			'size_restriction': None,
			}
formats['mpeg'] = {'label': 'MPEG-1',
			 'suffix': 'mpg',
			 'ffmpeg_name': 'mpeg1video',
			 'ffmpeg_codec': None,
			 'ffmpeg_quality': mpeg_quality,
			 'limited_framerates': True,
			 'size_restriction': None,
			 }
formats['wmv'] = {'label': 'WMV2',
			'suffix': 'wmv',
			'ffmpeg_name': 'asf',
			'ffmpeg_codec': 'wmv2',
			'ffmpeg_quality': vp8_quality,
			'limited_framerates': False,
			'size_restriction': None,
			}

movie_format_synonyms = {'webm': 'vp8',
			 'ogg': 'theora',
			 'ogv': 'theora',
			 'qt': 'mov',
			 'quicktime': 'mov'
			 }
for f2, f in movie_format_synonyms.items():
    formats[f2] = formats[f]
