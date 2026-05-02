from main_inverse_static import main


if __name__ == "__main__":
    print(
        "main_periodic_spring_adaptive.py is deprecated. "
        "Use main_inverse_static.py with ADAPTIVE_SPRING_ENABLED=YES and "
        "ADAPTIVE_SPRING_MODE='levels' or 'every_refine'."
    )
    main(
        forced_run_settings={
            "do_static": False,
            "do_adaptive_grad": False,
            "do_adaptive_spring": True,
        },
        forced_adaptive_spring_settings={
            "enabled": True,
        },
    )
