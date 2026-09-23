/**
 * A tiny, dependency-free schema builder that produces JSON Schema and
 * infers TypeScript types, plus a validator for model output and tool args.
 *
 *   const Weather = s.object({
 *     city: s.string(),
 *     tempC: s.number().describe("Temperature in Celsius"),
 *     conditions: s.enum(["sunny", "cloudy", "rain"]),
 *     note: s.string().optional(),
 *   });
 *   type Weather = Infer<typeof Weather>;
 */

import type { JSONSchema } from "./types.js";

export class Schema<T> {
  declare readonly _type: T;
  constructor(
    public readonly jsonSchema: JSONSchema,
    public readonly isOptional = false,
  ) {}

  describe(description: string): Schema<T> {
    return new Schema<T>({ ...this.jsonSchema, description }, this.isOptional);
  }

  optional(): Schema<T | undefined> {
    return new Schema<T | undefined>(this.jsonSchema, true);
  }

  nullable(): Schema<T | null> {
    const js = this.jsonSchema;
    if (typeof js.type === "string") return new Schema({ ...js, type: [js.type, "null"] }, this.isOptional);
    return new Schema({ anyOf: [js, { type: "null" }] }, this.isOptional);
  }

  default(value: T): Schema<T | undefined> {
    return new Schema<T | undefined>({ ...this.jsonSchema, default: value }, true);
  }
}

export type Infer<S> = S extends Schema<infer T> ? T : never;
type Shape = Record<string, Schema<any>>;
type OptionalKeys<S extends Shape> = { [K in keyof S]: S[K]["isOptional"] extends true ? K : undefined extends Infer<S[K]> ? K : never }[keyof S];
type ObjectOf<S extends Shape> = { [K in Exclude<keyof S, OptionalKeys<S>>]: Infer<S[K]> } & {
  [K in OptionalKeys<S>]?: Infer<S[K]>;
};

export const s = {
  string: (opts: { description?: string; minLength?: number; maxLength?: number; pattern?: string; format?: string } = {}) =>
    new Schema<string>({ type: "string", ...opts }),
  number: (opts: { description?: string; minimum?: number; maximum?: number } = {}) =>
    new Schema<number>({ type: "number", ...opts }),
  integer: (opts: { description?: string; minimum?: number; maximum?: number } = {}) =>
    new Schema<number>({ type: "integer", ...opts }),
  boolean: (opts: { description?: string } = {}) => new Schema<boolean>({ type: "boolean", ...opts }),
  enum: <const V extends readonly (string | number)[]>(values: V, opts: { description?: string } = {}) =>
    new Schema<V[number]>({
      enum: [...values],
      ...(values.every((v) => typeof v === "string") ? { type: "string" } : {}),
      ...opts,
    }),
  literal: <const V extends string | number | boolean>(value: V) => new Schema<V>({ const: value }),
  array: <I>(items: Schema<I>, opts: { description?: string; minItems?: number; maxItems?: number } = {}) =>
    new Schema<I[]>({ type: "array", items: items.jsonSchema, ...opts }),
  object: <S extends Shape>(shape: S, opts: { description?: string } = {}) => {
    const properties: Record<string, JSONSchema> = {};
    const required: string[] = [];
    for (const [k, v] of Object.entries(shape)) {
      properties[k] = v.jsonSchema;
      if (!v.isOptional) required.push(k);
    }
    return new Schema<ObjectOf<S>>({
      type: "object",
      properties,
      ...(required.length ? { required } : {}),
      additionalProperties: false,
      ...opts,
    });
  },
  record: <V>(values: Schema<V>) => new Schema<Record<string, V>>({ type: "object", additionalProperties: values.jsonSchema }),
  union: <A extends Schema<any>[]>(...options: A) => new Schema<Infer<A[number]>>({ anyOf: options.map((o) => o.jsonSchema) }),
  any: () => new Schema<unknown>({}),
  /** Wrap an existing JSON Schema. Supply the type parameter yourself. */
  json: <T = unknown>(schema: JSONSchema) => new Schema<T>(schema),
};

export function toJSONSchema(value: Schema<any> | JSONSchema | undefined): JSONSchema | undefined {
  if (value === undefined) return undefined;
  if (value instanceof Schema) return value.jsonSchema;
  return value;
}

// ---------------------------------------------------------------- strict mode

/** True when every object lists all properties as required and forbids extras (OpenAI strict mode). */
export function isStrictCompatible(schema: unknown): boolean {
  if (!schema || typeof schema !== "object") return true;
  const sc = schema as Record<string, any>;
  if (sc.type === "object" || "properties" in sc) {
    const props = Object.keys(sc.properties ?? {});
    if (sc.additionalProperties !== false) return false;
    const req = new Set<string>(sc.required ?? []);
    if (req.size !== props.length || !props.every((p) => req.has(p))) return false;
  }
  for (const sub of Object.values(sc.properties ?? {})) if (!isStrictCompatible(sub)) return false;
  for (const key of ["items", "additionalProperties"]) {
    if (sc[key] && typeof sc[key] === "object" && !isStrictCompatible(sc[key])) return false;
  }
  for (const key of ["anyOf", "oneOf", "allOf", "prefixItems"]) {
    for (const sub of sc[key] ?? []) if (!isStrictCompatible(sub)) return false;
  }
  return true;
}

// ---------------------------------------------------------------- validation

export class SchemaValidationError extends Error {}

function isType(value: unknown, t: string): boolean {
  switch (t) {
    case "string":
      return typeof value === "string";
    case "integer":
      return typeof value === "number" && Number.isInteger(value);
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "boolean":
      return typeof value === "boolean";
    case "array":
      return Array.isArray(value);
    case "object":
      return typeof value === "object" && value !== null && !Array.isArray(value);
    case "null":
      return value === null;
    default:
      return true;
  }
}

/** Validate against a practical subset of JSON Schema. Throws `SchemaValidationError`. */
export function validate(value: unknown, schema: JSONSchema | undefined, path = "$"): void {
  if (!schema || Object.keys(schema).length === 0) return;
  const sc = schema as Record<string, any>;
  const options = sc.anyOf ?? sc.oneOf;
  if (options) {
    const errors: string[] = [];
    const ok = options.some((opt: JSONSchema) => {
      try {
        validate(value, opt, path);
        return true;
      } catch (e) {
        errors.push((e as Error).message);
        return false;
      }
    });
    if (!ok) throw new SchemaValidationError(`${path}: does not match any allowed shape (${errors.slice(0, 3).join("; ")})`);
  }
  for (const sub of sc.allOf ?? []) validate(value, sub, path);
  if ("const" in sc && value !== sc.const) throw new SchemaValidationError(`${path}: expected ${JSON.stringify(sc.const)}`);
  if (sc.enum && !sc.enum.includes(value as never))
    throw new SchemaValidationError(`${path}: ${JSON.stringify(value)} is not one of ${JSON.stringify(sc.enum)}`);
  if (sc.type !== undefined) {
    const allowed: string[] = Array.isArray(sc.type) ? sc.type : [sc.type];
    if (!allowed.some((t) => isType(value, t))) {
      const got = value === null ? "null" : Array.isArray(value) ? "array" : typeof value;
      throw new SchemaValidationError(`${path}: expected ${allowed.join("/")}, got ${got}`);
    }
  }
  if (isType(value, "object")) {
    const obj = value as Record<string, unknown>;
    const props: Record<string, JSONSchema> = sc.properties ?? {};
    for (const req of sc.required ?? []) {
      if (!(req in obj)) throw new SchemaValidationError(`${path}: missing required property '${req}'`);
    }
    for (const [k, v] of Object.entries(obj)) {
      if (k in props) validate(v, props[k], `${path}.${k}`);
      else if (sc.additionalProperties === false) throw new SchemaValidationError(`${path}: unexpected property '${k}'`);
      else if (sc.additionalProperties && typeof sc.additionalProperties === "object")
        validate(v, sc.additionalProperties, `${path}.${k}`);
    }
  }
  if (Array.isArray(value)) {
    if (sc.prefixItems) sc.prefixItems.forEach((p: JSONSchema, i: number) => i < value.length && validate(value[i], p, `${path}[${i}]`));
    else if (sc.items && typeof sc.items === "object") value.forEach((v, i) => validate(v, sc.items, `${path}[${i}]`));
    if (sc.minItems !== undefined && value.length < sc.minItems) throw new SchemaValidationError(`${path}: needs at least ${sc.minItems} items`);
    if (sc.maxItems !== undefined && value.length > sc.maxItems) throw new SchemaValidationError(`${path}: allows at most ${sc.maxItems} items`);
  }
  if (typeof value === "string") {
    if (sc.minLength !== undefined && value.length < sc.minLength) throw new SchemaValidationError(`${path}: shorter than ${sc.minLength}`);
    if (sc.maxLength !== undefined && value.length > sc.maxLength) throw new SchemaValidationError(`${path}: longer than ${sc.maxLength}`);
    if (sc.pattern && !new RegExp(sc.pattern).test(value)) throw new SchemaValidationError(`${path}: does not match pattern ${sc.pattern}`);
  }
  if (typeof value === "number") {
    if (sc.minimum !== undefined && value < sc.minimum) throw new SchemaValidationError(`${path}: must be >= ${sc.minimum}`);
    if (sc.maximum !== undefined && value > sc.maximum) throw new SchemaValidationError(`${path}: must be <= ${sc.maximum}`);
  }
}

/** Parse JSON from model text, tolerating code fences and surrounding prose. */
export function extractJson(text: string): unknown {
  let str = text.trim();
  const fence = /```(?:json)?\s*([\s\S]*?)```/.exec(str);
  if (fence) str = fence[1].trim();
  try {
    return JSON.parse(str);
  } catch {
    /* fall through */
  }
  for (const [open, close] of [["{", "}"], ["[", "]"]]) {
    const start = str.indexOf(open);
    const end = str.lastIndexOf(close);
    if (start !== -1 && end > start) {
      try {
        return JSON.parse(str.slice(start, end + 1));
      } catch {
        /* next */
      }
    }
  }
  throw new SchemaValidationError("Response was not valid JSON");
}

/** Schema used for an agent's output type. Non-object schemas are wrapped in {"value": ...}. */
export function outputSchemaFor(schema: JSONSchema): { schema: JSONSchema; wrapped: boolean } {
  if (schema.type === "object") return { schema, wrapped: false };
  return {
    schema: { type: "object", properties: { value: schema }, required: ["value"], additionalProperties: false },
    wrapped: true,
  };
}
