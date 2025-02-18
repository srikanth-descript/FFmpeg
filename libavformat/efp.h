/*
 * EFP format
 * Copyright (c) 2024 FFmpeg.
 */

#ifndef AVFORMAT_EFP_H
#define AVFORMAT_EFP_H

#include <stdint.h>
#include "version.h"
#include "libavutil/opt.h"

#define EFP_MAGIC "EFP\0"
#define EFP_MAGIC_SIZE 4

typedef struct EFPContext {
    int64_t data_start;     /* Offset where data starts */
    int64_t packet_count;   /* Number of packets */
    int stream_index;       /* Current stream being processed */
} EFPContext;

static const AVOption efp_options[] = {
    { NULL }
};

static const AVClass efp_class = {
    .class_name = "efp",
    .item_name  = av_default_item_name,
    .option     = efp_options,
    .version    = LIBAVFORMAT_VERSION_INT,
};

static const AVCodecTag codec_efp_tags[] = {
    { AV_CODEC_ID_MPEG4,           MKTAG('m', 'p', '4', 'v') },
    { AV_CODEC_ID_H264,            MKTAG('a', 'v', 'c', '1') },
    { AV_CODEC_ID_H264,            MKTAG('a', 'v', 'c', '3') },
    { AV_CODEC_ID_HEVC,            MKTAG('h', 'e', 'v', '1') },
    { AV_CODEC_ID_HEVC,            MKTAG('h', 'v', 'c', '1') },
    { AV_CODEC_ID_HEVC,            MKTAG('d', 'v', 'h', '1') },
    { AV_CODEC_ID_VVC,             MKTAG('v', 'v', 'c', '1') },
    { AV_CODEC_ID_VVC,             MKTAG('v', 'v', 'i', '1') },
    { AV_CODEC_ID_EVC,             MKTAG('e', 'v', 'c', '1') },
    { AV_CODEC_ID_MPEG2VIDEO,      MKTAG('m', 'p', '4', 'v') },
    { AV_CODEC_ID_MPEG1VIDEO,      MKTAG('m', 'p', '4', 'v') },
    { AV_CODEC_ID_MJPEG,           MKTAG('m', 'p', '4', 'v') },
    { AV_CODEC_ID_PNG,             MKTAG('m', 'p', '4', 'v') },
    { AV_CODEC_ID_JPEG2000,        MKTAG('m', 'p', '4', 'v') },
    { AV_CODEC_ID_VC1,             MKTAG('v', 'c', '-', '1') },
    { AV_CODEC_ID_DIRAC,           MKTAG('d', 'r', 'a', 'c') },
    { AV_CODEC_ID_TSCC2,           MKTAG('m', 'p', '4', 'v') },
    { AV_CODEC_ID_VP9,             MKTAG('v', 'p', '0', '9') },
    { AV_CODEC_ID_AV1,             MKTAG('a', 'v', '0', '1') },
    { AV_CODEC_ID_AAC,             MKTAG('m', 'p', '4', 'a') },
    { AV_CODEC_ID_ALAC,            MKTAG('a', 'l', 'a', 'c') },
    { AV_CODEC_ID_MP4ALS,          MKTAG('m', 'p', '4', 'a') },
    { AV_CODEC_ID_MP3,             MKTAG('m', 'p', '4', 'a') },
    { AV_CODEC_ID_MP2,             MKTAG('m', 'p', '4', 'a') },
    { AV_CODEC_ID_AC3,             MKTAG('a', 'c', '-', '3') },
    { AV_CODEC_ID_EAC3,            MKTAG('e', 'c', '-', '3') },
    { AV_CODEC_ID_DTS,             MKTAG('m', 'p', '4', 'a') },
    { AV_CODEC_ID_TRUEHD,          MKTAG('m', 'l', 'p', 'a') },
    { AV_CODEC_ID_FLAC,            MKTAG('f', 'L', 'a', 'C') },
    { AV_CODEC_ID_OPUS,            MKTAG('O', 'p', 'u', 's') },
    { AV_CODEC_ID_VORBIS,          MKTAG('m', 'p', '4', 'a') },
    { AV_CODEC_ID_QCELP,           MKTAG('m', 'p', '4', 'a') },
    { AV_CODEC_ID_EVRC,            MKTAG('m', 'p', '4', 'a') },
    { AV_CODEC_ID_DVD_SUBTITLE,    MKTAG('m', 'p', '4', 's') },
    { AV_CODEC_ID_MOV_TEXT,        MKTAG('t', 'x', '3', 'g') },
    { AV_CODEC_ID_BIN_DATA,        MKTAG('g', 'p', 'm', 'd') },
    { AV_CODEC_ID_MPEGH_3D_AUDIO,  MKTAG('m', 'h', 'm', '1') },
    { AV_CODEC_ID_NONE,            MKTAG('m', 'p', '4', 'v') },
};

static const AVCodecTag *const codec_efp_tags_list[] = { codec_efp_tags, NULL};

#endif /* AVFORMAT_EFP_H */
