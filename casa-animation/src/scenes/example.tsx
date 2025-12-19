// final trim:
// ffmpeg -i full_res.mp4 -vf "crop=1200:1050:0:150" full_res_trimmed.mp4
// ffmpeg -i half_res.mp4 -vf "crop=600:525:0:75" half_res_trimmed.mp4
import { SVG, makeScene2D } from "@motion-canvas/2d";
import { createRef, waitFor, all, loop } from "@motion-canvas/core";
import {
  Rect,
  Circle,
  Txt,
  Layout,
  Line,
  Path,
} from "@motion-canvas/2d/lib/components";
import catFaceSvg from "../../assets/cat-svgrepo-com-3.svg?raw";
import catMeowsSvg from "../../assets/cat-svgrepo-com-2.svg?raw";
import houseSvg from "../../assets/house-window-svgrepo-com.svg?raw";

export default makeScene2D(function* (view) {
  // Set view size to reduce wasted space
  //view.fill("#0a0a0a");
  const frame_y = 300;
  const frame_y_down = frame_y + 200;
  const space_between_frames = 260;
  const start_x = -800;
  const sa_x = start_x + 50;
  const sa_y = frame_y - 450;
  const casa_x = start_x + 500;
  const casa_y = frame_y - 450;
  const output_x_shift = -70;
  const token_y = 65;
  // Refs
  const frames = [createRef<Rect>(), createRef<Rect>(), createRef<Rect>()];
  const frameCats = [
    createRef<Layout>(),
    createRef<Layout>(),
    createRef<Layout>(),
  ];
  const frameBorders = [
    createRef<Layout>(),
    createRef<Layout>(),
    createRef<Layout>(),
  ];
  const captions = [createRef<Txt>(), createRef<Txt>(), createRef<Txt>()];
  const legend = [
    createRef<Txt>(),
    createRef<Txt>(),
    createRef<Txt>(),
    createRef<Txt>(),
  ];
  const tokenGroups = [
    createRef<Layout>(),
    createRef<Layout>(),
    createRef<Layout>(),
  ];
  const tokenBorder = createRef<Rect>();
  const saBlock = createRef<Rect>();
  const casaBlock = createRef<Rect>();
  const saOutputs = [createRef<Rect>(), createRef<Rect>(), createRef<Rect>()];
  const casaOutputs = [createRef<Rect>(), createRef<Rect>(), createRef<Rect>()];
  const downArrows = [createRef<Line>(), createRef<Line>(), createRef<Line>()];
  const upArrows = [createRef<Line>(), createRef<Line>(), createRef<Line>()];
  const mergedCats = [
    createRef<Layout>(),
    createRef<Layout>(),
    createRef<Layout>(),
  ];
  const mergedBorders = [
    createRef<Layout>(),
    createRef<Layout>(),
    createRef<Layout>(),
  ];

  const colors = ["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b"];
  const sentence = "A little gray cat - is meowing at - my house door".split(
    "-"
  );

  const legendSize = 35;

  const createMeow = (ref: any) => (
    <Layout ref={ref} opacity={1}>
      <SVG svg={catMeowsSvg} width={150} lineWidth={10}></SVG>
    </Layout>
  );

  const createHouse = (ref: any) => (
    <Layout ref={ref} opacity={0.78}>
      <SVG svg={houseSvg} width={140}></SVG>
    </Layout>
  );
  // Cat shape function
  const createCat = (ref: any) => (
    <Layout ref={ref} opacity={1}>
      <SVG svg={catFaceSvg} width={150}></SVG>
    </Layout>
  );

  const createBorders = (ref: any) => (
    <Layout ref={ref} opacity={1}>
      <Rect width={8} height={140} fill="rgba(0,0,0,0.4)" x={-86} />
      <Rect width={8} height={140} fill="rgba(0,0,0,0.4)" x={86} />
    </Layout>
  );

  // Initial video frames with captions - centered better
  for (let i = 0; i < 3; i++) {
    const captionText = sentence[i];

    view.add(
      <>
        <Rect
          ref={frames[i]}
          width={180}
          height={140}
          fill={colors[i]}
          x={start_x + i * space_between_frames}
          y={frame_y}
          radius={12}
          shadowBlur={10}
          shadowColor="rgba(0,0,0,0.3)"
        >
          {i == 0
            ? createCat(frameCats[i])
            : i == 1
            ? createMeow(frameCats[i])
            : createHouse(frameCats[i])}
          {createBorders(frameBorders[i])}
        </Rect>
        <Txt
          ref={captions[i]}
          text={captionText}
          fill="#fff"
          fontSize={26}
          fontFamily="Arial"
          fontWeight={600}
          x={start_x + i * space_between_frames}
          y={frame_y - 100}
        />
      </>
    );
  }

  yield* waitFor(1);

  // Transform captions into tokens and shift frames up
  for (let i = 0; i < 3; i++) {
    const tokenLayout = (
      <Layout
        ref={tokenGroups[i]}
        x={start_x - 35 + i * space_between_frames}
        y={token_y}
        opacity={0}
        direction="row"
      >
        {[...Array(3)].map((_, j) => (
          <Rect
            width={30}
            height={30}
            fill={colors[i]}
            x={j * 38}
            radius={4}
            stroke="#fff"
            lineWidth={2}
          />
        ))}
      </Layout>
    );
    view.add(tokenLayout);
  }
  view.add(
    <Rect
      ref={tokenBorder}
      width={0}
      height={170}
      stroke="rgb(89,174,149)"
      lineWidth={3}
      radius={8}
      opacity={0}
      x={sa_x}
      y={token_y}
    />
  );

  // Add L2 block (above LAYER) - moved further right
  view.add(
    <Rect
      ref={saBlock}
      width={250}
      height={180}
      fill="rgba(89,174,149,0)"
      x={sa_x}
      y={sa_y}
      radius={12}
      shadowBlur={15}
      shadowColor="rgb(89,174,149)"
      stroke="rgb(89,174,149)"
      lineWidth={2}
      opacity={0}
    >
      <Txt
        text="Self-Attention"
        fill="rgb(89,174,149)"
        fontSize={32}
        fontWeight={600}
      />
    </Rect>
  );

  // Add LAYER block (below L2) - moved further right
  view.add(
    <Rect
      ref={casaBlock}
      width={250}
      height={180}
      fill="rgba(244,185,91, 0)"
      x={casa_x}
      y={casa_y}
      radius={12}
      shadowBlur={15}
      shadowColor="rgb(244,185,91)"
      stroke="rgb(244,185,91)"
      lineWidth={2}
      opacity={0}
    >
      <Txt text="CASA" fill="rgb(244,185,91)" fontSize={32} fontWeight={600} />
    </Rect>
  );

  yield* all(
    ...frames.map((f) => f().position.y(frame_y_down, 1)),
    ...frames.map((f) => f().opacity(0.3, 1)),
    ...tokenGroups.map((t) => t().opacity(0.3, 1)),
    ...captions.map((f) => f().opacity(0.3, 1)),
    saBlock().opacity(0.3, 1),
    casaBlock().opacity(0.3, 1)
  );

  yield* waitFor(0.5);

  // Add all L2 output tokens but hidden initially
  for (let i = 0; i < 3; i++) {
    const tokenLayout = (
      <Layout
        ref={saOutputs[i]}
        x={start_x - 110 + (i - 1) * 120}
        y={sa_y - 200}
        opacity={0}
        direction="row"
      >
        {[...Array(3)].map((_, j) => (
          <Rect
            width={30}
            height={30}
            fill={colors[i]}
            radius={4}
            stroke="rgb(89,174,149)"
            shadowBlur={8}
            shadowColor="rgb(89,174,149)"
            lineWidth={5}
            x={j * 38}
            opacity={11}
          />
        ))}
      </Layout>
    );
    view.add(tokenLayout);
  }

  // Add all L2 output tokens but hidden initially
  for (let i = 0; i < 3; i++) {
    const tokenLayout = (
      <Layout
        ref={casaOutputs[i]}
        x={casa_x - 150 + (i - 1) * 120}
        y={casa_y - 200}
        opacity={0}
        direction="row"
      >
        {[...Array(3)].map((_, j) => (
          <Rect
            width={30}
            height={30}
            fill={colors[i]}
            radius={4}
            stroke="rgb(244,185,91)"
            shadowBlur={8}
            shadowColor="rgb(244,185,91)"
            lineWidth={5}
            x={j * 38}
            opacity={11}
          />
        ))}

        {i == 0
          ? createCat(mergedCats[i])
          : i == 1
          ? createMeow(mergedCats[i])
          : createHouse(mergedCats[i])}
        {createBorders(mergedBorders[i])}
      </Layout>
    );
    view.add(tokenLayout);
    mergedCats[i]().opacity(0);
    mergedBorders[i]().opacity(0);
  }

  // Add arrows (hidden initially)
  for (let i = 0; i < 3; i++) {
    view.add(
      <Line
        ref={downArrows[i]}
        points={[
          [sa_x + output_x_shift + 2 * 66, sa_y - 250],
          [sa_x + output_x_shift + 3 * 66 + 50, sa_y - 250],
        ]}
        stroke="#64748b"
        lineWidth={3}
        endArrow
        arrowSize={8}
        opacity={0}
        lineCap="round"
      />
    );
    view.add(
      <Line
        ref={upArrows[i]}
        points={[
          [casa_x + output_x_shift - 20, casa_y - 250],
          [casa_x + output_x_shift - 126, casa_y - 250],
        ]}
        stroke="#64748b"
        lineWidth={3}
        endArrow
        arrowSize={8}
        opacity={0}
        lineCap="round"
      />
    );
  }

  // Legend
  view.add(
    <Txt
      ref={legend[0]}
      text="Visual tokens"
      fill="#fff"
      fontSize={legendSize}
      fontFamily="Arial"
      fontWeight={600}
      x={0}
      y={frame_y + 200}
      opacity={0}
    />
  );
  view.add(
    <Txt
      ref={legend[1]}
      text="Video captions"
      fill="#fff"
      fontSize={legendSize}
      fontFamily="Arial"
      fontWeight={600}
      x={0}
      y={frame_y - 100}
      opacity={0}
    />
  );
  view.add(
    <Txt
      ref={legend[2]}
      text="Text tokens"
      fill="#fff"
      fontSize={legendSize}
      fontFamily="Arial"
      fontWeight={600}
      x={0}
      y={token_y}
      opacity={0}
    />
  );
  view.add(
    <Txt
      ref={legend[3]}
      text="Output tokens"
      fill="#fff"
      fontSize={legendSize}
      fontFamily="Arial"
      fontWeight={600}
      x={0}
      y={casa_y - 200}
      opacity={0}
    />
  );

  yield* all(...legend.map((a) => a().opacity(1, 0.8)));
  yield* waitFor(0.8);
  yield* all(...legend.map((a) => a().opacity(0, 0.8)));
  yield* waitFor(0.8);

  // Process each token group one by one
  for (let i = 0; i < 3; i++) {
    // FIRST ANIMATION: Focus tokens, L2 opaque, LAYER translucid, show L2 output
    const opacityPromises1 = tokenGroups.map((t, idx) => {
      if (idx <= i) return t().opacity(1, 0.5);
      return t().opacity(0.3, 0.5);
    });

    yield* all(
      ...opacityPromises1,
      saBlock().opacity(1, 0.5),
      casaBlock().opacity(0.3, 0.5),
      tokenBorder().opacity(1, 0.5),
      tokenBorder().width((i / 2) * space_between_frames + 150 * (i + 1), 0.5),
      tokenBorder().position.x(start_x + (i / 2) * space_between_frames, 0.5),
      captions[i]().opacity(1, 0.5),
      saOutputs[i]().opacity(1, 0.5),
      saOutputs[i]().position.x(saOutputs[i]().position.x() + 120, 0.5)
    );
    yield* waitFor(0.2);

    // Little pause in between
    const opacityPromisesInter = tokenGroups.map((t, idx) => {
      return t().opacity(0.3, 0.8);
    });
    yield* all(
      ...opacityPromisesInter,
      tokenBorder().opacity(0, 0.8),
      saBlock().opacity(0.3, 0.8)
    );
    yield* waitFor(0.4);

    // SECOND ANIMATION: Focus current group only, LAYER opaque, L2 translucid, frame goes down
    const opacityPromises2 = tokenGroups.map((t, idx) => {
      if (idx === i) return t().opacity(1, 0.5);
      return t().opacity(0.3, 0.5);
    });

    yield* all(
      ...opacityPromises2,
      frames[i]().position.y(frame_y, 0.8),
      frames[i]().opacity(1, 0.8),
      tokenBorder().opacity(0, 0.5),
      saBlock().opacity(0.3, 0.5),
      casaBlock().opacity(1, 0.5),
      casaOutputs[i]().opacity(1, 0.5),
      casaOutputs[i]().position.x(casaOutputs[i]().position.x() + 120, 0.5)
    );

    yield* waitFor(0.4);

    yield* all(
      frames[i]().position.y(frame_y_down, 0.8),
      frames[i]().opacity(0.3, 0.8),
      casaBlock().opacity(0.3, 0.5)
    );

    yield* waitFor(0.2);
  }

  // FINAL ANIMATION: Everything becomes translucid except outputs
  yield* all(
    ...frames.map((f) => f().opacity(0.2, 0.8)),
    ...captions.map((f) => f().opacity(0.2, 0.8)),
    ...tokenGroups.map((t) => t().opacity(0.2, 0.8)),
    saBlock().opacity(0.2, 0.8),
    casaBlock().opacity(0.2, 0.8),
    ...downArrows.map((a) => a().opacity(1, 1.8)),
    ...upArrows.map((a) => a().opacity(1, 1.8))
  );

  yield* waitFor(0.3);

  // Move outputs to merge
  const mergeX = 230;
  yield* all(
    ...saOutputs.map((o) =>
      all(
        o().position.x(o().position.x() + mergeX, 1),
        ...o()
          .children()
          .map((child) => child.stroke("#ffffff", 0.8))
      )
    ),
    ...casaOutputs.map((o) =>
      all(
        o().position.x(o().position.x() - mergeX, 1),
        ...o()
          .children()
          .filter(function (n) {
            return n.hasOwnProperty("stroke");
          })
          .map((child) => child.stroke("#ffffff", 0.8))
      )
    ),
    ...downArrows.map((a) => a().opacity(0, 0.8)),
    ...upArrows.map((a) => a().opacity(0, 0.8))
  );

  yield* waitFor(0.5);

  // Focus on merged tokens and start growing them
  yield* all(...saOutputs.map((o) => all(o().opacity(0, 0.8))));

  // Smooth morph: grow tokens while moving and morphing into frames
  yield* all(
    // Grow and move tokens
    ...casaOutputs.map((o, i) =>
      all(
        o().position.x(start_x + i * space_between_frames, 1.5),
        o().position.y(frame_y, 1.5),
        o().scale(1, 1.5),
        // Morph each child token
        ...o()
          .children()
          .filter(function (n) {
            return n.hasOwnProperty("stroke");
          })
          .map((child, j) =>
            all(
              child.width(180, 1.5),
              child.height(140, 1.5),
              child.radius(12, 1.5),
              child.position.x(0, 1.5), // Center them back
              child.stroke("rgba(0,0,0,0)", 1.5),
              child.shadowBlur(0, 0.8),
              child.lineWidth(0, 0.8)
            )
          )
      )
    ),
    // Fade out everything else
    ...captions.map((f) => f().opacity(0, 1.2)),
    ...frames.map((f) => f().opacity(0, 1.2)),
    ...tokenGroups.map((t) => t().opacity(0, 1.2)),
    saBlock().opacity(0, 1.2),
    casaBlock().opacity(0, 1.2)
  );

  yield* waitFor(0.3);

  // Fade in cats and borders
  yield* all(
    ...mergedCats.map((c) => c().opacity(1, 0.8)),
    ...mergedBorders.map((b) => b().opacity(1, 0.8)),
    ...captions.map((c) => c().opacity(1, 0.8))
  );

  yield* waitFor(1);
});
