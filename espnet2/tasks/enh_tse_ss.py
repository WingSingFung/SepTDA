import argparse
from typing import Callable, Collection, Dict, List, Optional, Tuple
from pathlib import Path
import logging

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.enh.espnet_model_tse_ss import ESPnetExtractionEnhancementModel
from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.extractor.septda_extractor import SepformerTDAExtractor
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.enh import (
    criterion_choices,
    decoder_choices,
    encoder_choices,
    loss_wrapper_choices,
)
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import (
    AbsPreprocessor,
    TSEPreprocessor,
    EnhTsePreprocessor,
)
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

# added
from espnet2.train.distributed_utils import (
    DistributedOption,
)
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from espnet2.tasks.abs_task import IteratorOptions
from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.chunk_iter_factory import ChunkIterFactory
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.train.dataset import ESPnetDataset
from espnet2.tasks.abs_task import build_batch_sampler

extractor_choices = ClassChoices(
    name="extractor",
    classes=dict(
        septda=SepformerTDAExtractor,
    ),
    type_check=AbsExtractor,
    default="td_speakerbeam",
)

preprocessor_choices = ClassChoices(
    name="preprocessor",
    classes=dict(
        tse=TSEPreprocessor,
        enh_tse=EnhTsePreprocessor,
    ),
    type_check=AbsPreprocessor,
    default="tse",
)

MAX_REFERENCE_NUM = 100


class TargetSpeakerExtractionAndEnhancementTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    class_choices_list = [
        # --encoder and --encoder_conf
        encoder_choices,
        # --extractor and --extractor_conf
        extractor_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default="kaiming_uniform",
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetExtractionEnhancementModel),
            help="The keyword arguments for model class.",
        )

        group.add_argument(
            "--criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "si_snr",
                    "conf": {},
                    "wrapper": "fixed_order",
                    "wrapper_conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--train_spk2enroll",
            type=str_or_none,
            default=None,
            help="The scp file containing the mapping from speakerID to enrollment\n"
            "(This is used to sample the target-speaker enrollment signal)",
        )
        group.add_argument(
            "--enroll_segment",
            type=int_or_none,
            default=None,
            help="Truncate the enrollment audio to the specified length if not None",
        )
        group.add_argument(
            "--load_spk_embedding",
            type=str2bool,
            default=False,
            help="Whether to load speaker embeddings instead of enrollments",
        )
        group.add_argument(
            "--load_all_speakers",
            type=str2bool,
            default=False,
            help="Whether to load target-speaker for all speakers in each sample",
        )
        # inherited from EnhPreprocessor
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of signal-to-noise ratio (SNR) level in decibel.",
        )
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
        )
        group.add_argument(
            "--speech_volume_normalize",
            type=str_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value or range. "
            "e.g. --speech_volume_normalize 1.0 scales it to 1.0.\n"
            "--speech_volume_normalize 0.5_1.0 scales it to a random number in "
            "the range [0.5, 1.0)",
        )
        group.add_argument(
            "--use_reverberant_ref",
            type=str2bool,
            default=False,
            help="Whether to use reverberant speech references "
            "instead of anechoic ones",
        )
        group.add_argument(
            "--num_spk",
            type=int,
            default=1,
            help="Number of speakers in the input signal.",
        )
        group.add_argument(
            "--num_noise_type",
            type=int,
            default=1,
            help="Number of noise types.",
        )
        group.add_argument(
            "--sample_rate",
            type=int,
            default=8000,
            help="Sampling rate of the data (in Hz).",
        )
        group.add_argument(
            "--force_single_channel",
            type=str2bool,
            default=False,
            help="Whether to force all data to be single-channel.",
        )
        # used for selection {num_spk}-mix in 2-5mix
        group.add_argument(
            "--n_mix",
            type=int,
            nargs="+",
            default=None,
        )
        group.add_argument(
            "--task",
            type=str,
            default="enh",
            help="Used for preprocessor.",
        )
        group.add_argument(
            "--remove_samples_with_speaker_overlap",
            type=str2bool,
            default=False,
            help="Whether to remove data with speaker overlap (for WSJ0_Nmix)",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_iter_options(
        cls,
        args: argparse.Namespace,
        distributed_option: DistributedOption,
        mode: str,
    ):
        if mode == "train":
            preprocess_fn = cls.build_preprocess_fn(args, train=True)
            collate_fn = cls.build_collate_fn(args, train=True)
            data_path_and_name_and_type = (
                args.train_data_path_and_name_and_type
            )
            shape_files = args.train_shape_file
            batch_size = args.batch_size
            batch_bins = args.batch_bins
            batch_type = args.batch_type
            max_cache_size = args.max_cache_size
            max_cache_fd = args.max_cache_fd
            distributed = distributed_option.distributed
            num_batches = None
            num_iters_per_epoch = args.num_iters_per_epoch
            train = True

        elif mode == "valid":
            preprocess_fn = cls.build_preprocess_fn(args, train=False)
            collate_fn = cls.build_collate_fn(args, train=False)
            data_path_and_name_and_type = (
                args.valid_data_path_and_name_and_type
            )
            shape_files = args.valid_shape_file

            if args.valid_batch_type is None:
                batch_type = args.batch_type
            else:
                batch_type = args.valid_batch_type
            if args.valid_batch_size is None:
                batch_size = args.batch_size
            else:
                batch_size = args.valid_batch_size
            if args.valid_batch_bins is None:
                batch_bins = args.batch_bins
            else:
                batch_bins = args.valid_batch_bins
            if args.valid_max_cache_size is None:
                # Cache 5% of maximum size for validation loader
                max_cache_size = 0.05 * args.max_cache_size
            else:
                max_cache_size = args.valid_max_cache_size
            max_cache_fd = args.max_cache_fd
            distributed = distributed_option.distributed
            num_batches = None
            # num_iters_per_epoch = None
            num_iters_per_epoch = args.num_iters_valid  # added
            train = False

        elif mode == "plot_att":
            preprocess_fn = cls.build_preprocess_fn(args, train=False)
            collate_fn = cls.build_collate_fn(args, train=False)
            data_path_and_name_and_type = (
                args.valid_data_path_and_name_and_type
            )
            shape_files = args.valid_shape_file
            batch_type = "unsorted"
            batch_size = 1
            batch_bins = 0
            num_batches = args.num_att_plot
            max_cache_fd = args.max_cache_fd
            # num_att_plot should be a few sample ~ 3, so cache all data.
            max_cache_size = np.inf if args.max_cache_size != 0.0 else 0.0
            # always False because plot_attention performs on RANK0
            distributed = False
            num_iters_per_epoch = None
            train = False
        else:
            raise NotImplementedError(f"mode={mode}")

        return IteratorOptions(
            preprocess_fn=preprocess_fn,
            collate_fn=collate_fn,
            data_path_and_name_and_type=data_path_and_name_and_type,
            shape_files=shape_files,
            batch_type=batch_type,
            batch_size=batch_size,
            batch_bins=batch_bins,
            num_batches=num_batches,
            max_cache_size=max_cache_size,
            max_cache_fd=max_cache_fd,
            distributed=distributed,
            num_iters_per_epoch=num_iters_per_epoch,
            train=train,
        )

    @classmethod
    def build_chunk_iter_factory(
        cls,
        args: argparse.Namespace,
        iter_options: IteratorOptions,
        mode: str,
    ) -> AbsIterFactory:
        assert check_argument_types()

        dataset = ESPnetDataset(
            iter_options.data_path_and_name_and_type,
            float_dtype=args.train_dtype,
            preprocess=iter_options.preprocess_fn,
            max_cache_size=iter_options.max_cache_size,
            max_cache_fd=iter_options.max_cache_fd,
        )
        cls.check_task_requirements(
            dataset, args.allow_variable_data_keys, train=iter_options.train
        )

        if len(iter_options.shape_files) == 0:
            key_file = iter_options.data_path_and_name_and_type[0][0]
        else:
            key_file = iter_options.shape_files[0]

        batch_sampler = UnsortedBatchSampler(
            batch_size=1,
            key_file=key_file,
            # utt2category_file=utt2category_file,
        )
        batches = list(batch_sampler)

        # use only N-mix data if specified
        if args.n_mix is not None and mode != "plot_att":
            assert isinstance(args.n_mix, list)
            logging.info("Remove unspecified data")
            num_batches_before = len(batches)
            new_batches = []
            logging.info(args.n_mix)
            for batch in batches:
                if int(batch[0][0]) in args.n_mix:
                    new_batches.append(batch)
            batches = new_batches
            num_batches_after = len(batches)
            logging.info(
                f"Original batches: {num_batches_before}, Current batches: {num_batches_after}"
            )
        if iter_options.num_batches is not None:
            batches = batches[: iter_options.num_batches]
        logging.info(f"[{mode}] dataset:\n{dataset}")

        if iter_options.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            if len(batches) < world_size:
                raise RuntimeError(
                    "Number of samples is smaller than world_size"
                )
            if iter_options.batch_size < world_size:
                raise RuntimeError(
                    "batch_size must be equal or more than world_size"
                )

            if rank < iter_options.batch_size % world_size:
                batch_size = iter_options.batch_size // world_size + 1
            else:
                batch_size = iter_options.batch_size // world_size
            num_cache_chunks = args.num_cache_chunks // world_size
            # NOTE(kamo): Split whole corpus by sample numbers without considering
            #   each of the lengths, therefore the number of iteration counts are not
            #   always equal to each other and the iterations are limitted
            #   by the fewest iterations.
            #   i.e. the samples over the counts are discarded.
            batches = batches[rank::world_size]
        else:
            batch_size = iter_options.batch_size
            num_cache_chunks = args.num_cache_chunks

        return ChunkIterFactory(
            dataset=dataset,
            batches=batches,
            seed=args.seed,
            batch_size=batch_size,
            # For chunk iterator,
            # --num_iters_per_epoch doesn't indicate the number of iterations,
            # but indicates the number of samples.
            num_samples_per_epoch=iter_options.num_iters_per_epoch,
            shuffle=iter_options.train,
            num_workers=args.num_workers,
            collate_fn=iter_options.collate_fn,
            pin_memory=args.ngpu > 0,
            chunk_length=args.chunk_length,
            chunk_shift_ratio=args.chunk_shift_ratio,
            num_cache_chunks=num_cache_chunks,
            excluded_key_prefixes=args.chunk_excluded_key_prefixes,
        )

    @classmethod
    def build_sequence_iter_factory(
        cls, args: argparse.Namespace, iter_options: IteratorOptions, mode: str
    ) -> AbsIterFactory:
        assert check_argument_types()

        dataset = ESPnetDataset(
            iter_options.data_path_and_name_and_type,
            float_dtype=args.train_dtype,
            preprocess=iter_options.preprocess_fn,
            max_cache_size=iter_options.max_cache_size,
            max_cache_fd=iter_options.max_cache_fd,
        )
        cls.check_task_requirements(
            dataset, args.allow_variable_data_keys, train=iter_options.train
        )

        if Path(
            Path(iter_options.data_path_and_name_and_type[0][0]).parent,
            "utt2category",
        ).exists():
            utt2category_file = str(
                Path(
                    Path(
                        iter_options.data_path_and_name_and_type[0][0]
                    ).parent,
                    "utt2category",
                )
            )
            logging.info("\n\nReading " + utt2category_file)
        else:
            logging.info("\n\nNOT Reading utt2category" + utt2category_file)
            utt2category_file = None

        batch_sampler = build_batch_sampler(
            type=iter_options.batch_type,
            shape_files=iter_options.shape_files,
            fold_lengths=args.fold_length,
            batch_size=iter_options.batch_size,
            batch_bins=iter_options.batch_bins,
            sort_in_batch=args.sort_in_batch,
            sort_batch=args.sort_batch,
            drop_last=False,
            min_batch_size=torch.distributed.get_world_size()
            if iter_options.distributed
            else 1,
            utt2category_file=utt2category_file,
            remove_samples_with_speaker_overlap=args.remove_samples_with_speaker_overlap,
        )

        batches = list(batch_sampler)
        if iter_options.num_batches is not None:
            batches = batches[: iter_options.num_batches]
        if args.n_mix is not None and mode != "plot_att":
            assert isinstance(args.n_mix, list)
            logging.info("Remove unspecified data")
            num_batches_before = len(batches)
            new_batches = []
            logging.info(args.n_mix)
            for batch in batches:
                if int(batch[0][0]) in args.n_mix:
                    new_batches.append(batch)
            batches = new_batches
            num_batches_after = len(batches)
            logging.info(
                f"Original batches: {num_batches_before}, Current batches: {num_batches_after}"
            )

        bs_list = [len(batch) for batch in batches]

        logging.info(f"[{mode}] dataset:\n{dataset}")
        logging.info(f"[{mode}] Batch sampler: {batch_sampler}")
        logging.info(
            f"[{mode}] mini-batch sizes summary: N-batch={len(bs_list)}, "
            f"mean={np.mean(bs_list):.1f}, min={np.min(bs_list)}, max={np.max(bs_list)}"
        )
        if iter_options.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            for batch in batches:
                if len(batch) < world_size:
                    raise RuntimeError(
                        f"The batch-size must be equal or more than world_size: "
                        f"{len(batch)} < {world_size}"
                    )
            batches = [batch[rank::world_size] for batch in batches]

        return SequenceIterFactory(
            dataset=dataset,
            batches=batches,
            seed=args.seed,
            num_iters_per_epoch=iter_options.num_iters_per_epoch,
            shuffle=iter_options.train,
            num_workers=args.num_workers,
            collate_fn=iter_options.collate_fn,
            pin_memory=args.ngpu > 0,
        )

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()

        return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        # retval = TSEPreprocessor(
        retval = EnhTsePreprocessor(
            train=train,
            task=getattr(args, "task", "enh_tse"),
            dummy_label=getattr(args, "dummy_label", "dummy"),
            speech_segment=getattr(args, "chunk_length", None),
            # inherited from TSEPreprocessor
            train_spk2enroll=args.train_spk2enroll,
            enroll_segment=getattr(args, "enroll_segment", None),
            load_spk_embedding=getattr(args, "load_spk_embedding", False),
            load_all_speakers=getattr(args, "load_all_speakers", False),
            # inherited from EnhPreprocessor
            rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
            rir_apply_prob=args.rir_apply_prob
            if hasattr(args, "rir_apply_prob")
            else 1.0,
            noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
            noise_apply_prob=args.noise_apply_prob
            if hasattr(args, "noise_apply_prob")
            else 1.0,
            noise_db_range=args.noise_db_range
            if hasattr(args, "noise_db_range")
            else "13_15",
            short_noise_thres=args.short_noise_thres
            if hasattr(args, "short_noise_thres")
            else 0.5,
            speech_volume_normalize=args.speech_volume_normalize
            if hasattr(args, "speech_volume_normalize")
            else None,
            use_reverberant_ref=args.use_reverberant_ref
            if hasattr(args, "use_reverberant_ref")
            else None,
            num_spk=args.num_spk if hasattr(args, "num_spk") else 1,
            num_noise_type=args.num_noise_type
            if hasattr(args, "num_noise_type")
            else 1,
            sample_rate=args.sample_rate
            if hasattr(args, "sample_rate")
            else 8000,
            force_single_channel=args.force_single_channel
            if hasattr(args, "force_single_channel")
            else False,
        )
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = (
                "speech_mix",
                "enroll_ref1",
                "speech_ref1",
                "enroll_ref2",
                "speech_ref2",
            )
        else:
            # Inference mode
            retval = ("speech_mix", "enroll_ref1")
            # retval = ("speech_mix", )
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = [
            "enroll_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)
        ]
        if "speech_ref1" in retval:
            retval += [
                "speech_ref{}".format(n)
                for n in range(2, MAX_REFERENCE_NUM + 1)
            ]
        else:
            retval += [
                "speech_ref{}".format(n)
                for n in range(1, MAX_REFERENCE_NUM + 1)
            ]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(
        cls, args: argparse.Namespace
    ) -> ESPnetExtractionEnhancementModel:
        assert check_argument_types()
        encoder = encoder_choices.get_class(args.encoder)(**args.encoder_conf)
        extractor = extractor_choices.get_class(args.extractor)(
            encoder.output_dim, **args.extractor_conf
        )
        decoder = decoder_choices.get_class(args.decoder)(**args.decoder_conf)

        loss_wrappers = []
        if getattr(args, "criterions", None) is not None:
            # This check is for the compatibility when load models
            # that packed by older version
            for ctr in args.criterions:
                criterion_conf = ctr.get("conf", {})
                criterion = criterion_choices.get_class(ctr["name"])(
                    **criterion_conf
                )
                loss_wrapper = loss_wrapper_choices.get_class(ctr["wrapper"])(
                    criterion=criterion, **ctr["wrapper_conf"]
                )
                loss_wrappers.append(loss_wrapper)

        # 1. Build model
        model = ESPnetExtractionEnhancementModel(
            encoder=encoder,
            extractor=extractor,
            decoder=decoder,
            loss_wrappers=loss_wrappers,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
