from main_inverse_static import main


if __name__ == "__main__":
    main(
        forced_run_settings={
            "do_static": False,
            "do_adaptive_grad": False,
            "do_adaptive_spring": True,
        },
        forced_adaptive_spring_settings={
            "enabled": True,
            "mode": "final",
        },
    )
