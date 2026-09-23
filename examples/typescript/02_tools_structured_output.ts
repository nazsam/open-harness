// Tools plus typed, validated output.
import { Agent, run, s, tool, type Infer } from "openharness";

const getWeather = tool({
  name: "get_weather",
  description: "Get the current weather for a city.",
  parameters: s.object({ city: s.string().describe('City name, e.g. "Toronto".') }),
  execute: ({ city }) => {
    const fake: Record<string, [number, string]> = { toronto: [14, "cloudy"], paris: [21, "sunny"], london: [11, "rain"] };
    const [tempC, conditions] = fake[city.toLowerCase()] ?? [18, "cloudy"];
    return { city, tempC, conditions };
  },
});

const TripAdvice = s.object({
  city: s.string(),
  tempC: s.number(),
  conditions: s.enum(["sunny", "cloudy", "rain"]),
  pack: s.array(s.string()),
});
type TripAdvice = Infer<typeof TripAdvice>;

const agent = new Agent<TripAdvice>({
  name: "Travel helper",
  instructions: "Check the weather with your tool, then suggest what to pack.",
  tools: [getWeather],
  outputType: TripAdvice,
});

const result = await run(agent, "I'm flying to London tomorrow.");
const advice: TripAdvice = result.output;
console.log(advice);
console.log(`${result.turns} turns, ${result.usage.totalTokens} tokens`);
